# SPDX-FileCopyrightText: 2026 Pier Luigi Fiorini
# SPDX-License-Identifier: MIT

if(NOT COMMAND CPMAddPackage)
    set(cpm_file "${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM_0.40.2.cmake")
    set(cpm_sha256 "c8cdc32c03816538ce22781ed72964dc864b2a34a310d3b7104812a5ca2d835d")
    if(EXISTS "${cpm_file}")
        file(SHA256 "${cpm_file}" existing_hash)
    endif()
    if(NOT existing_hash STREQUAL cpm_sha256)
        file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/cmake")
        file(DOWNLOAD
            "https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.40.2/CPM.cmake"
            "${cpm_file}"
            EXPECTED_HASH "SHA256=${cpm_sha256}"
            TLS_VERIFY ON
            STATUS download_status)
        list(GET download_status 0 download_code)
        if(NOT download_code EQUAL 0)
            message(FATAL_ERROR "Could not download CPM.cmake: ${download_status}")
        endif()
    endif()
    include("${cpm_file}")
endif()

# Use the core's pinned Eigen submodule. DOWNLOAD_ONLY avoids the core's own
# tools and upstream directory-wide flags.
CPMAddPackage(
    NAME nam_loader_dependency
    GIT_REPOSITORY https://github.com/sdatkinson/NeuralAmpModelerCore.git
    GIT_TAG 1f42f88535884450104b8711d7595019afa0495b
    GIT_SUBMODULES Dependencies/eigen
    DOWNLOAD_ONLY YES)

# JSON is vendored in NAM rather than a submodule. Fetch the matching release
# from its own repository and pin the commit behind the v3.12.0 tag.
CPMAddPackage(
    NAME nam_loader_json
    VERSION 3.12.0
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG 55f93686c01528224f448c19128836e7df245f72
    DOWNLOAD_ONLY YES)

# Only the header-only ResamplingContainer (Lanczos + WDL resampler) is used by
# ChainDomain.h; the commit has no submodule that ResamplingContainer needs
# (its Dependencies/ holds vendored WDL and Lanczos headers). CPM cannot pass an
# empty GIT_SUBMODULES, so the unused Eigen submodule of AudioDSPTools is still
# cloned once into the CPM cache; DOWNLOAD_ONLY keeps its tools out of the build.
CPMAddPackage(
    NAME nam_loader_audiodsptools
    GIT_REPOSITORY https://github.com/sdatkinson/AudioDSPTools.git
    GIT_TAG 0827c6c2fc0deced568536142ea86f189e0b98a1
    DOWNLOAD_ONLY YES)

# The DSP sources include JUCE modules. A parent project (e.g. Ampira, which
# patches JUCE at configure time) must win, so JUCE is only fetched when no
# one has added it already.
if(NOT COMMAND juce_add_module)
    CPMAddPackage(
        NAME JUCE
        GITHUB_REPOSITORY juce-framework/JUCE
        GIT_TAG 9.0.3
        VERSION 9.0.3)
endif()

if(NAM_LOADER_BUILD_TESTS AND NOT TARGET GTest::gtest)
    CPMAddPackage(
        NAME GOOGLETEST
        GITHUB_REPOSITORY google/googletest
        GIT_TAG v1.15.2
        VERSION 1.15.2
        OPTIONS
            "INSTALL_GTEST OFF")
endif()
