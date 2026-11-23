// Copyright 2026 Jonas Teuwen. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "aifocore/utilities/thread_pool_singleton.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <functional>

#if !defined(_WIN32)
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace aifocore {
namespace {

// Runs `count` tasks on the shared pool; returns how many ran.
int RunTasks(int count) {
  std::atomic<int> ran{0};
  ThreadPoolManager::GetInstance()
      .submit_sequence(
          0, count, [&](int) { ran.fetch_add(1, std::memory_order_relaxed); })
      .wait();
  return ran.load();
}

class ThreadPoolManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    initial_threads_ = ThreadPoolManager::GetInstance().get_thread_count();
  }

  void TearDown() override {
    ThreadPoolManager::SetThreadCount(initial_threads_);
  }

  std::size_t initial_threads_ = 0;
};

TEST_F(ThreadPoolManagerTest, RunsTasks) {
  EXPECT_GE(ThreadPoolManager::GetInstance().get_thread_count(), 1U);
  EXPECT_EQ(RunTasks(100), 100);
}

TEST_F(ThreadPoolManagerTest, ReturnsTheSamePool) {
  EXPECT_EQ(&ThreadPoolManager::GetInstance(),
            &ThreadPoolManager::GetInstance());
}

TEST_F(ThreadPoolManagerTest, SetThreadCountResizesInPlace) {
  InlineThreadPool* before = &ThreadPoolManager::GetInstance();
  ThreadPoolManager::SetThreadCount(3);
  EXPECT_EQ(&ThreadPoolManager::GetInstance(), before);
  EXPECT_EQ(ThreadPoolManager::GetInstance().get_thread_count(), 3U);
  EXPECT_EQ(RunTasks(50), 50);
}

#if !defined(_WIN32)

// Forks, runs `child` in the child with a 20 s alarm (a hang kills it with
// SIGALRM), and expects the child to exit with status 0. `child` returns the
// exit status; with `use_exit` the child leaves through std::exit, so static
// destructors (and the pool's) run, otherwise through _exit.
void ExpectChildSucceeds(const std::function<int()>& child,
                         bool use_exit = false) {
  const pid_t pid = fork();
  ASSERT_NE(pid, -1);
  if (pid == 0) {
    alarm(20);
    const int status = child();
    if (use_exit) {
      std::exit(status);
    }
    _exit(status);
  }
  int status = 0;
  ASSERT_EQ(waitpid(pid, &status, 0), pid);
  ASSERT_TRUE(WIFEXITED(status))
      << "child killed by signal "
      << (WIFSIGNALED(status) ? WTERMSIG(status) : 0)
      << " (SIGALRM = " << SIGALRM << " means it hung)";
  EXPECT_EQ(WEXITSTATUS(status), 0);
}

TEST_F(ThreadPoolManagerTest, ChildAfterForkGetsAWorkingPool) {
  ThreadPoolManager::SetThreadCount(4);
  ASSERT_EQ(RunTasks(64), 64);  // the parent's threads exist and have run
  ExpectChildSucceeds([] {
    if (RunTasks(64) != 64) {
      return 1;
    }
    // Same size as in the parent.
    return ThreadPoolManager::GetInstance().get_thread_count() == 4 ? 0 : 2;
  });
  EXPECT_EQ(RunTasks(64), 64);  // the parent's pool is unaffected
}

TEST_F(ThreadPoolManagerTest, ChildAfterForkCanResizeThePool) {
  ASSERT_EQ(RunTasks(16), 16);
  ExpectChildSucceeds([] {
    ThreadPoolManager::SetThreadCount(2);
    if (ThreadPoolManager::GetInstance().get_thread_count() != 2) {
      return 1;
    }
    return RunTasks(32) == 32 ? 0 : 2;
  });
}

TEST_F(ThreadPoolManagerTest, ChildAfterForkExitsNormally) {
  ASSERT_EQ(RunTasks(16), 16);
  // std::exit runs static destructors: the child must not join the threads
  // it inherited (they do not exist there).
  ExpectChildSucceeds([] { return RunTasks(16) == 16 ? 0 : 1; },
                      /*use_exit=*/true);
}

TEST_F(ThreadPoolManagerTest, ChildAfterForkExitsWithoutUsingThePool) {
  ASSERT_EQ(RunTasks(16), 16);
  ExpectChildSucceeds([] { return 0; }, /*use_exit=*/true);
}

#endif  // !defined(_WIN32)

}  // namespace
}  // namespace aifocore
