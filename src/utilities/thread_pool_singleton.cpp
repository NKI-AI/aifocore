// Copyright 2025 Jonas Teuwen. All Rights Reserved.
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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <mutex>

#include "aifocore/utilities/bs_thread_pool.h"

#if !defined(_WIN32) && !defined(__EMSCRIPTEN__)
#include <pthread.h>
#define AIFOCORE_THREAD_POOL_HANDLES_FORK 1
#endif

namespace aifocore {

namespace {

std::size_t GetThreadCountFromEnv() {
  const char* env_value = std::getenv("NUM_THREADS");
  if (env_value == nullptr) {
    return 0;  // Not set, use default
  }

  // Try to parse as integer
  char* end = nullptr;
  const int64_t value = std::strtol(env_value, &end, 10);

  // Check if parsing was successful and value is reasonable
  if (end == env_value || *end != '\0' || value < 0) {
    // Invalid value, use default
    return 0;
  }

  // Return the parsed value (0 will be interpreted as hardware_concurrency by
  // BS::thread_pool)
  return static_cast<std::size_t>(value);
}

// The process-wide pool.
//
// A thread pool does not survive fork(): the child gets a copy of the pool
// object but none of its threads, so a task submitted there never runs, and
// destroying the copy would join threads that do not exist. A fork handler
// therefore drops the copy in the child (it is leaked on purpose), and the
// next GetInstance() in the child creates a new pool with the parent's
// number of threads.
struct PoolState {
  // Guards creating, resizing and replacing the pool. Held across fork(), so
  // the child never sees it locked by a thread that does not exist there.
  std::mutex mutex;
  std::atomic<InlineThreadPool*> pool{nullptr};
  // Thread count for the next pool created (0 = NUM_THREADS or hardware
  // concurrency). Guarded by `mutex`.
  std::size_t next_thread_count = 0;

  PoolState() = default;
  PoolState(const PoolState&) = delete;
  PoolState& operator=(const PoolState&) = delete;

  // At normal exit, stop and join the pool's threads, as the function-local
  // static pool did before.
  ~PoolState() { delete pool.load(std::memory_order_acquire); }
};

PoolState& State();

#ifdef AIFOCORE_THREAD_POOL_HANDLES_FORK
void LockBeforeFork() {
  State().mutex.lock();
}

void UnlockInParentAfterFork() {
  State().mutex.unlock();
}

void ResetInChildAfterFork() {
  PoolState& state = State();
  InlineThreadPool* inherited = state.pool.load(std::memory_order_relaxed);
  if (inherited != nullptr) {
    state.next_thread_count = inherited->get_thread_count();
    state.pool.store(nullptr, std::memory_order_relaxed);
  }
  state.mutex.unlock();
}
#endif

PoolState& State() {
  static PoolState state;
#ifdef AIFOCORE_THREAD_POOL_HANDLES_FORK
  static const bool fork_handlers_registered = [] {
    return pthread_atfork(&LockBeforeFork, &UnlockInParentAfterFork,
                          &ResetInChildAfterFork) == 0;
  }();
  static_cast<void>(fork_handlers_registered);
#endif
  return state;
}

// Creates the pool. Requires `state.mutex` and no current pool.
InlineThreadPool& CreatePoolLocked(PoolState& state, std::size_t count) {
  auto* pool = new InlineThreadPool(count);
  state.pool.store(pool, std::memory_order_release);
  return *pool;
}

}  // namespace

InlineThreadPool& ThreadPoolManager::GetInstance() {
  return GetPool(0);
}

void ThreadPoolManager::SetThreadCount(std::size_t count) {
  PoolState& state = State();
  std::lock_guard<std::mutex> lock(state.mutex);
  state.next_thread_count = count;
  InlineThreadPool* pool = state.pool.load(std::memory_order_relaxed);
  if (pool == nullptr) {
    CreatePoolLocked(state, count);
    return;
  }
  pool->reset(count);
}

InlineThreadPool& ThreadPoolManager::GetPool(std::size_t count) {
  PoolState& state = State();
  if (InlineThreadPool* pool = state.pool.load(std::memory_order_acquire)) {
    return *pool;
  }
  std::lock_guard<std::mutex> lock(state.mutex);
  if (InlineThreadPool* pool = state.pool.load(std::memory_order_relaxed)) {
    return *pool;
  }
  // On first call, check environment variable for thread count override
  static const std::size_t env_thread_count = GetThreadCountFromEnv();
  if (count == 0) {
    count = state.next_thread_count != 0 ? state.next_thread_count
                                         : env_thread_count;
  }
  return CreatePoolLocked(state, count);
}

}  // namespace aifocore
