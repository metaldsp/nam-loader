# SPDX-FileCopyrightText: 2026 TONE3000
# SPDX-FileCopyrightText: 2026 Pier Luigi Fiorini
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.24)
# C is required since JUCE 9 (juce_graphics and other modules include C sources).
project(NamLoader VERSION 0.1.0 LANGUAGES C CXX)

option(NAM_LOADER_BUILD_TESTS "Build the standalone library tests" OFF)
option(NAM_LOADER_BUILD_EXAMPLE "Build the model inspection example" OFF)

include(cmake/Dependencies.cmake)

foreach(required
        "${nam_loader_dependency_SOURCE_DIR}/NAM/get_dsp.h"
        "${nam_loader_dependency_SOURCE_DIR}/Dependencies/eigen/Eigen/Core"
        "${nam_loader_json_SOURCE_DIR}/single_include/nlohmann/json.hpp"
        "${nam_loader_audiodsptools_SOURCE_DIR}/dsp/ResamplingContainer/ResamplingContainer.h")
    if(NOT EXISTS "${required}")
        message(FATAL_ERROR "A downloaded NamLoader dependency is missing ${required}")
    endif()
endforeach()

set(core "${nam_loader_dependency_SOURCE_DIR}/NAM")
add_library(nam_loader_core STATIC
    ${core}/activations.cpp
    ${core}/conv1d.cpp
    ${core}/container.cpp
    ${core}/convnet.cpp
    ${core}/dsp.cpp
    ${core}/get_dsp.cpp
    ${core}/linear.cpp
    ${core}/lstm.cpp
    ${core}/ring_buffer.cpp
    ${core}/util.cpp
    ${core}/wavenet/a2_fast.cpp
    ${core}/wavenet/model.cpp
    ${core}/wavenet/slimmable.cpp)
target_include_directories(nam_loader_core SYSTEM PUBLIC
    "${nam_loader_dependency_SOURCE_DIR}"
    "${nam_loader_dependency_SOURCE_DIR}/Dependencies/eigen"
    "${nam_loader_json_SOURCE_DIR}/single_include/nlohmann")
target_compile_definitions(nam_loader_core PUBLIC NAM_ENABLE_A2_FAST)
target_compile_features(nam_loader_core PUBLIC cxx_std_20)
if(WIN32)
    target_compile_definitions(nam_loader_core PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# nam_loader is an INTERFACE library on purpose: the JUCE module sources are
# compiled into the final consumer target, so a STATIC wrapper would compile
# JUCE a second time (duplicate symbols / ODR violations). The consumer
# compiles these sources with its own JUCE configuration instead.
add_library(nam_loader INTERFACE)
add_library(NamLoader::NamLoader ALIAS nam_loader)
target_sources(nam_loader INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Engine.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/RateBoundary.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Runtime.cpp")
target_include_directories(nam_loader INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
# Only src/RateBoundary.cpp includes AudioDSPTools' ResamplingContainer (and,
# through it, the WDL/Lanczos headers). The paths are INTERFACE because the
# sources compile in the consumer; the public headers never include them.
target_include_directories(nam_loader SYSTEM INTERFACE
    "${nam_loader_audiodsptools_SOURCE_DIR}"
    "${nam_loader_audiodsptools_SOURCE_DIR}/dsp/ResamplingContainer/Dependencies")
target_compile_features(nam_loader INTERFACE cxx_std_20)
if(WIN32)
    # AudioDSPTools uses std::min/std::max.
    target_compile_definitions(nam_loader INTERFACE NOMINMAX)
endif()
# ConfigParserHelper registers architectures in otherwise unreferenced objects.
# Link the core whole-archive so the registrations survive in the consumer.
# The library only needs juce_audio_basics (AudioWorkgroup, Thread,
# SystemStats, AudioBuffer, LinearSmoothedValue).
target_link_libraries(nam_loader INTERFACE
    juce::juce_audio_basics
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,nam_loader_core>")

set_target_properties(nam_loader_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_EXTENSIONS OFF
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
# Keep registries private to each plugin binary loaded into the same host.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(nam_loader_core PRIVATE -fno-gnu-unique)
endif()

if(NAM_LOADER_BUILD_EXAMPLE)
    juce_add_console_app(nam_loader_inspect PRODUCT_NAME "nam_loader_inspect")
    target_sources(nam_loader_inspect PRIVATE examples/inspect.cpp)
    target_compile_definitions(nam_loader_inspect PRIVATE
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0)
    target_link_libraries(nam_loader_inspect PRIVATE
        NamLoader::NamLoader
        juce::juce_recommended_config_flags)
endif()

if(NAM_LOADER_BUILD_TESTS)
    enable_testing()

    juce_add_console_app(NamLoaderTests PRODUCT_NAME "NamLoaderTests")
    file(GLOB nam_loader_test_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")
    target_sources(NamLoaderTests PRIVATE ${nam_loader_test_sources})
    target_compile_definitions(NamLoaderTests PRIVATE
        JUCE_WEB_BROWSER=0
        JUCE_USE_WEBP=0
        JUCE_USE_CURL=0
        # Prevent windows.h min/max macros from breaking std::min/std::max.
        $<$<PLATFORM_ID:Windows>:NOMINMAX>
        # Absolute fixture paths (tests/test_helpers.h), so the binary runs
        # from anywhere: captures from Ampira's suite, and the core's models.
        NAM_LOADER_TEST_FILES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/files"
        NAM_LOADER_EXAMPLE_MODELS_DIR="${nam_loader_dependency_SOURCE_DIR}/example_models")
    target_link_libraries(NamLoaderTests
        PRIVATE
            NamLoader::NamLoader
            juce::juce_dsp
            juce::juce_audio_formats
            GTest::gtest
        PUBLIC
            juce::juce_recommended_config_flags)

    include(GoogleTest)
    # PRE_TEST discovers at ctest time, so a build never has to run the binary.
    gtest_discover_tests(NamLoaderTests DISCOVERY_MODE PRE_TEST)
endif()
