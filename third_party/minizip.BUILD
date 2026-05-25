load("@rules_cc//cc:defs.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
)

SOURCE_FILES = [
    "mz_crypt.c",
    "mz_crypt_openssl.c",
    "mz_os.c",
    "mz_os_posix.c",
    "mz_strm.c",
    "mz_strm_buf.c",
    "mz_strm_bzip.c",
    "mz_strm_lzma.c",
    "mz_strm_mem.c",
    "mz_strm_os_posix.c",
    "mz_strm_split.c",
    "mz_strm_zlib.c",
    "mz_strm_zstd.c",
    "mz_zip.c",
    "compat/ioapi.c",
    "compat/unzip.c",
    "compat/zip.c",
] + select({
    "@platforms//os:macos": [
        "mz_crypt_apple.c",
        "mz_strm_libcomp.c",
    ],
    "//conditions:default": [],
})

HEADER_FILES = [
    "compat/crypt.h",
    "compat/ioapi.h",
    "compat/unzip.h",
    "compat/zip.h",
    "mz_crypt.h",
    "mz.h",
    "mz_os.h",
    "mz_strm_buf.h",
    "mz_strm_bzip.h",
    "mz_strm.h",
    "mz_strm_lzma.h",
    "mz_strm_mem.h",
    "mz_strm_os.h",
    "mz_strm_split.h",
    "mz_strm_zlib.h",
    "mz_strm_zstd.h",
    "mz_zip.h",
    "mz_zip_rw.h",
] + select({
    "@platforms//os:macos": [
        "mz_strm_libcomp.h",
    ],
    "//conditions:default": [],
})

cc_library(
    name = "minizip",
    srcs = SOURCE_FILES,
    hdrs = HEADER_FILES,
    copts = [
        "-DMZ_ZLIB",
        "-DZLIB_COMPAT",
        "-DMZ_COMPAT",
        "-DMZ_BZIP2",
        "-DMZ_LZMA",
        "-DMZ_ZSTD",
        "-D_FILE_OFFSET_BITS=64",
    ],
    includes = [
        ".",
        "compat",
    ],
    linkopts = [
        "-lz",
        "-pthread",
        "-ldl",
    ],
    deps = [
        "@bzip2//:bz2",
        "@openssl//:crypto",
        "@openssl//:openssl",
        "@openssl//:ssl",
        "@xz//:lzma",
        "@zlib",
        "@zstd",
    ],
)
