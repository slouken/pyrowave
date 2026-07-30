cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_STANDARD 99)
project(pyrowave LANGUAGES CXX C)

if (CMAKE_COMPILER_IS_GNUCXX OR (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang"))
    set(PYROWAVE_CXX_FLAGS -Wshadow -Wall -Wextra -Wno-comment -Wno-missing-field-initializers -Wno-empty-body -fvisibility=hidden)
    if (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
        set(PYROWAVE_CXX_FLAGS ${PYROWAVE_CXX_FLAGS} -Wno-backslash-newline-escape)
    endif()
    if (NOT (${CMAKE_BUILD_TYPE} MATCHES "Release"))
        message("Enabling frame pointer for profiling/debug.")
        set(PYROWAVE_CXX_FLAGS ${PYROWAVE_CXX_FLAGS} -fno-omit-frame-pointer)
    endif()
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
        message("Enabling SSE3 support.")
        set(PYROWAVE_CXX_FLAGS ${PYROWAVE_CXX_FLAGS} -msse3)
    endif()
elseif (MSVC)
    set(PYROWAVE_CXX_FLAGS /D_CRT_SECURE_NO_WARNINGS /wd4267 /wd4244 /wd4309 /wd4005 /MP)
endif()

add_library(pyrowave STATIC
        pyrowave_config.hpp shaders/slangmosh.hpp
        pyrowave_encoder.hpp pyrowave_encoder.cpp
        pyrowave_decoder.hpp pyrowave_decoder.cpp
        pyrowave_common.hpp pyrowave_common.cpp)

target_include_directories(pyrowave PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(pyrowave PRIVATE ${PYROWAVE_CXX_FLAGS})
set_target_properties(pyrowave PROPERTIES POSITION_INDEPENDENT_CODE ON)

option(PYROWAVE_DEVEL "Build PyroWave as a standalone sandbox with development tooling." OFF)

if (ANDROID)
    # Mobile chips really like FP16
    target_compile_definitions(pyrowave PRIVATE PYROWAVE_PRECISION=0)
    message("Forcing FP16 default on Android.")
else()
    option(PYROWAVE_FP32_STORAGE "Configure for full FP32 and extended precision." OFF)
    option(PYROWAVE_FP32_MATH "Configure for FP32 arithmetic, but reduced range storage." ON)
    if (PYROWAVE_FP32_STORAGE)
        target_compile_definitions(pyrowave PRIVATE PYROWAVE_PRECISION=2)
    elseif(PYROWAVE_FP32_MATH)
        target_compile_definitions(pyrowave PRIVATE PYROWAVE_PRECISION=1)
    else()
        target_compile_definitions(pyrowave PRIVATE PYROWAVE_PRECISION=0)
    endif()
endif()

if (PYROWAVE_DEVEL)
    add_library(pyrowave-utils STATIC yuv4mpeg.cpp yuv4mpeg.hpp)
    target_compile_options(pyrowave-utils PRIVATE ${PYROWAVE_CXX_FLAGS})
    target_include_directories(pyrowave-utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

    # Standalone build, build the sandbox.
    set(GRANITE_RENDERER ON CACHE BOOL "" FORCE)
    set(GRANITE_VULKAN_FOSSILIZE OFF CACHE BOOL "" FORCE)
    set(GRANITE_SHIPPING OFF CACHE BOOL "" FORCE)
    set(GRANITE_PLATFORM "SDL" CACHE STRING "" FORCE)
    set(GRANITE_HIDDEN ON CACHE BOOL "" FORCE)
    set(GRANITE_POSITION_INDEPENDENT ON CACHE BOOL "" FORCE)
    set(GRANITE_FFMPEG ON CACHE BOOL "" FORCE)
    set(GRANITE_FFMPEG_VULKAN ON CACHE BOOL "" FORCE)
    add_subdirectory(Granite EXCLUDE_FROM_ALL)

    add_granite_offline_tool(pyrowave-sandbox sandbox.cpp)
    target_link_libraries(pyrowave-sandbox PRIVATE pyrowave)
    target_compile_definitions(pyrowave-sandbox PRIVATE ASSET_DIRECTORY=\"${CMAKE_CURRENT_SOURCE_DIR}/shaders\")
    target_link_libraries(pyrowave-sandbox PRIVATE granite-renderer pyrowave-utils)

    add_granite_offline_tool(pyrowave-bench bench.cpp)
    target_link_libraries(pyrowave-bench PRIVATE pyrowave)
    target_link_libraries(pyrowave-bench PRIVATE granite-vulkan pyrowave-utils)

    add_granite_offline_tool(pyrowave-encode encode.cpp)
    target_link_libraries(pyrowave-encode PRIVATE pyrowave)
    target_link_libraries(pyrowave-encode PRIVATE granite-vulkan pyrowave-utils)

    add_granite_offline_tool(pyrowave-decode decode.cpp)
    target_link_libraries(pyrowave-decode PRIVATE pyrowave)
    target_link_libraries(pyrowave-decode PRIVATE granite-vulkan pyrowave-utils)

    add_granite_offline_tool(pyrowave-psnr psnr.cpp)
    target_link_libraries(pyrowave-psnr PRIVATE pyrowave-utils)

    add_granite_application(pyrowave-viewer viewer.cpp)
    target_link_libraries(pyrowave-viewer PRIVATE pyrowave pyrowave-utils)
    if (NOT ANDROID)
        target_compile_definitions(pyrowave-viewer PRIVATE ASSET_DIRECTORY=\"${CMAKE_CURRENT_SOURCE_DIR}/shaders\")
    endif()

    add_granite_application(pyrowave-evaluator evaluator.cpp)
    target_link_libraries(pyrowave-evaluator PRIVATE granite-video pyrowave-utils pyrowave)
    if (NOT ANDROID)
        target_compile_definitions(pyrowave-evaluator PRIVATE ASSET_DIRECTORY=\"${CMAKE_CURRENT_SOURCE_DIR}/shaders\")
    endif()
elseif (${PROJECT_IS_TOP_LEVEL} AND (NOT TARGET granite-vulkan))
    set(GRANITE_RENDERER OFF CACHE BOOL "" FORCE)
    set(GRANITE_VULKAN_SPIRV_CROSS OFF CACHE BOOL "" FORCE)
    set(GRANITE_VULKAN_SHADER_MANAGER_RUNTIME_COMPILER OFF CACHE BOOL "" FORCE)
    set(GRANITE_VULKAN_SYSTEM_HANDLES OFF CACHE BOOL "" FORCE)
    set(GRANITE_SHADER_COMPILER_OPTIMIZE OFF CACHE BOOL "" FORCE)
    set(GRANITE_POSITION_INDEPENDENT ON CACHE BOOL "" FORCE)
    set(GRANITE_VULKAN_FOSSILIZE OFF CACHE BOOL "" FORCE)
    set(GRANITE_SHIPPING ON CACHE BOOL "" FORCE)
    set(GRANITE_PLATFORM "null" CACHE STRING "" FORCE)
    add_subdirectory(Granite EXCLUDE_FROM_ALL)
endif()

target_link_libraries(pyrowave PRIVATE granite-vulkan granite-math)

if (${PROJECT_IS_TOP_LEVEL})
    add_library(pyrowave-shared SHARED pyrowave_c.cpp pyrowave.h)
    target_link_libraries(pyrowave-shared PRIVATE pyrowave granite-vulkan)
    target_compile_definitions(pyrowave-shared PRIVATE PYROWAVE_EXPORT_SYMBOLS)
    target_compile_options(pyrowave-shared PRIVATE ${PYROWAVE_CXX_FLAGS})

    add_executable(pyrowave-c-test pyrowave_c_test.cpp)
    target_link_libraries(pyrowave-c-test PRIVATE pyrowave-shared granite-volk-headers)
    target_compile_options(pyrowave-c-test PRIVATE ${PYROWAVE_CXX_FLAGS})

    add_executable(pyrowave-c-interop-test pyrowave_c_interop_test.cpp com_ptr.hpp)
    target_link_libraries(pyrowave-c-interop-test PRIVATE pyrowave-shared granite-vulkan)
    target_compile_options(pyrowave-c-interop-test PRIVATE ${PYROWAVE_CXX_FLAGS})

    add_executable(pyrowave-device-validation pyrowave_device_validation.cpp com_ptr.hpp)
    target_link_libraries(pyrowave-device-validation PRIVATE pyrowave-shared granite-util granite-volk-headers)
    target_compile_options(pyrowave-device-validation PRIVATE ${PYROWAVE_CXX_FLAGS})

    set(PYROWAVE_API_VERSION_MAJOR 0)
    set(PYROWAVE_API_VERSION_MINOR 4)
    set(PYROWAVE_API_VERSION_PATCH 0)
    set(PYROWAVE_API_VERSION ${PYROWAVE_API_VERSION_MAJOR}.${PYROWAVE_API_VERSION_MINOR}.${PYROWAVE_API_VERSION_PATCH})

    # DLL_NAME_WITH_SOVERSION requires CMake 3.27+. SteamRT doesn't have CMake 3.27 yet though.
    if (WIN32 AND (CMAKE_VERSION VERSION_LESS "3.27"))
        message(FATAL_ERROR "CMake build on Windows expects version 3.27 at minimum.")
    endif()

    set_target_properties(pyrowave-shared
            PROPERTIES VERSION ${PYROWAVE_API_VERSION}
            SOVERSION ${PYROWAVE_API_VERSION_MAJOR}
            DLL_NAME_WITH_SOVERSION ON)

    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/pyrowave.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/pyrowave)
    install(TARGETS pyrowave-shared EXPORT pyrowave-sharedConfig
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/pyrowave)
    install(EXPORT pyrowave-sharedConfig DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pyrowave-shared/cmake)
    install(TARGETS pyrowave-device-validation RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

    set(PYROWAVE_INSTALL_LIB_DIR ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
    set(PYROWAVE_INSTALL_INC_DIR ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/pyrowave)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pkg-config/pyrowave-shared.pc.in
            ${CMAKE_CURRENT_BINARY_DIR}/pyrowave-shared.pc @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/pyrowave-shared.pc DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)

    if (WIN32 AND CMAKE_COMPILER_IS_GNUCXX)
        target_link_libraries(pyrowave-shared PRIVATE -static gcc stdc++ winpthread)
        target_link_libraries(pyrowave-c-test PRIVATE -static gcc stdc++ winpthread)
        target_link_libraries(pyrowave-c-interop-test PRIVATE -static gcc stdc++ winpthread)
        target_link_libraries(pyrowave-device-validation PRIVATE -static gcc stdc++ winpthread)
    endif()

    if (NOT WIN32)
        target_link_libraries(pyrowave-shared PRIVATE -pthread)
        target_link_libraries(pyrowave-c-test PRIVATE -pthread)
        target_link_libraries(pyrowave-c-interop-test PRIVATE -pthread)
        target_link_libraries(pyrowave-device-validation PRIVATE -pthread)
    endif()

    if (WIN32)
        target_sources(pyrowave-shared PRIVATE pyrowave-shared.def)
        set_target_properties(pyrowave-shared PROPERTIES PREFIX "lib")
        target_link_libraries(pyrowave-c-interop-test PRIVATE d3d11 d3d12 dxgi)
    elseif(NOT APPLE)
        target_link_options(pyrowave-shared PRIVATE -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/link.T)
    endif()
endif()

# Platform independent test for the Granite-free CPU bitstream layer.
if (${PROJECT_IS_TOP_LEVEL})
    add_executable(pyrowave-bitstream-test pyrowave_bitstream_test.cpp pyrowave_bitstream.cpp)
    target_include_directories(pyrowave-bitstream-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(pyrowave-bitstream-test PRIVATE ${PYROWAVE_CXX_FLAGS})
    target_compile_features(pyrowave-bitstream-test PRIVATE cxx_std_17)
endif()

########
# Metal decoder backend, for Apple Silicon (Apple7 GPU family and up).
#
# This is a decode-only alternative to the Vulkan path and shares no targets with
# it: the two backends are never linked into the same binary. Nothing here depends
# on Granite, which does not build on macOS, so the Metal targets can be built on
# their own with e.g. `ninja pyrowave-metal-decode`.

if (APPLE)
    set(PYROWAVE_METAL_DEFAULT ON)
else()
    set(PYROWAVE_METAL_DEFAULT OFF)
endif()

option(PYROWAVE_METAL "Build the Metal decoder backend (Apple only)." ${PYROWAVE_METAL_DEFAULT})

if (PYROWAVE_METAL AND NOT APPLE)
    message(FATAL_ERROR "PYROWAVE_METAL requires an Apple platform.")
endif()

if (PYROWAVE_METAL)
    # The MSL in shaders/metal is generated by shaders/metal/transpile.sh and
    # committed, so building does not require glslang or spirv-cross.
    add_library(pyrowave-metal STATIC
            pyrowave_metal.h
            pyrowave_metal_common.mm pyrowave_metal_common.hpp
            pyrowave_decoder_metal.mm
            pyrowave_encoder_metal.mm
            pyrowave_bitstream.cpp pyrowave_bitstream.hpp
            shaders/metal/pyrowave_msl.h)

    target_include_directories(pyrowave-metal PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(pyrowave-metal PRIVATE ${PYROWAVE_CXX_FLAGS})
    # The backends are Objective-C++ under ARC, so the id<MTL...> members in the
    # internal header are strong references and need no explicit retain or release.
    target_compile_options(pyrowave-metal PRIVATE -fobjc-arc)
    # The project as a whole is C++14, but this backend's C++ needs C++17. PUBLIC so
    # that anything linking it is raised to C++17 as well.
    target_compile_features(pyrowave-metal PUBLIC cxx_std_17)
    set_target_properties(pyrowave-metal PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(pyrowave-metal PUBLIC
            # IOSurface is for the encoder's GPU input path, which wraps the
            # caller's surfaces itself.
            "-framework Metal" "-framework Foundation" "-framework QuartzCore"
            "-framework IOSurface" "-framework CoreFoundation")

    if (${PROJECT_IS_TOP_LEVEL})
        # yuv4mpeg is otherwise only built for the Granite sandbox, which is
        # unavailable here, so define it if the Vulkan side has not already.
        if (NOT TARGET pyrowave-utils)
            add_library(pyrowave-utils STATIC yuv4mpeg.cpp yuv4mpeg.hpp)
            target_compile_options(pyrowave-utils PRIVATE ${PYROWAVE_CXX_FLAGS})
            target_include_directories(pyrowave-utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
        endif()

        add_executable(pyrowave-metal-decode pyrowave_metal_decode.mm)
        target_link_libraries(pyrowave-metal-decode PRIVATE pyrowave-metal pyrowave-utils)
        target_compile_options(pyrowave-metal-decode PRIVATE ${PYROWAVE_CXX_FLAGS} -fobjc-arc)

        add_executable(pyrowave-metal-encode pyrowave_metal_encode.mm)
        target_link_libraries(pyrowave-metal-encode PRIVATE pyrowave-metal pyrowave-utils)
        target_compile_options(pyrowave-metal-encode PRIVATE ${PYROWAVE_CXX_FLAGS} -fobjc-arc)

        add_executable(pyrowave-metal-test pyrowave_metal_test.mm)
        target_link_libraries(pyrowave-metal-test PRIVATE pyrowave-metal)
        target_compile_options(pyrowave-metal-test PRIVATE ${PYROWAVE_CXX_FLAGS} -fobjc-arc)

        # SDL viewer. Needs the per plane IOSurface texture properties, so it only
        # builds against an SDL that has them.
        set(PYROWAVE_SDL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Granite/third_party/sdl3)
        if (EXISTS ${PYROWAVE_SDL_DIR}/CMakeLists.txt)
            set(PYROWAVE_METAL_VIEWER_DEFAULT ON)
        else()
            set(PYROWAVE_METAL_VIEWER_DEFAULT OFF)
        endif()
        option(PYROWAVE_METAL_VIEWER "Build the SDL viewer for the Metal decoder."
                ${PYROWAVE_METAL_VIEWER_DEFAULT})

        if (PYROWAVE_METAL_VIEWER)
            if (NOT TARGET SDL3::SDL3)
                add_subdirectory(${PYROWAVE_SDL_DIR} sdl3 EXCLUDE_FROM_ALL)
            endif()

            add_executable(pyrowave-metal-viewer pyrowave_metal_viewer.mm)
            target_link_libraries(pyrowave-metal-viewer PRIVATE
                    pyrowave-metal SDL3::SDL3 "-framework IOSurface")
            target_compile_options(pyrowave-metal-viewer PRIVATE ${PYROWAVE_CXX_FLAGS} -fobjc-arc)
        endif()

        install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/pyrowave_metal.h
                DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/pyrowave)
    endif()
endif()

