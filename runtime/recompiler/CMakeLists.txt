# eden_shader: el recompilador de shaders de Eden (third_party/eden, GPL-3) como librería estática,
# solo con el backend SPIR-V. De su common/ usa las cabeceras; settings/logging/assert se sustituyen
# por las de shim/, que van antes en el include path.
#
# Lo usan tools/shadertool (en el Mac) y libnnsdk.so (en la Flip). Hace GPL-3 a lo que lo enlaza.

set(TP ${CMAKE_CURRENT_LIST_DIR}/../../third_party)

set(FMT_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(${TP}/fmt ${CMAKE_BINARY_DIR}/fmt EXCLUDE_FROM_ALL)

set(SIRIT_TESTS OFF CACHE BOOL "" FORCE)
set(SIRIT_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(${TP}/sirit ${CMAKE_BINARY_DIR}/sirit EXCLUDE_FROM_ALL)

file(GLOB_RECURSE EDEN_SHADER_SOURCES ${TP}/eden/src/shader_recompiler/*.cpp)
list(FILTER EDEN_SHADER_SOURCES EXCLUDE REGEX "/backend/(glsl|glasm)/")
add_library(eden_shader STATIC ${EDEN_SHADER_SOURCES})
target_compile_features(eden_shader PUBLIC cxx_std_20)
target_include_directories(eden_shader BEFORE PUBLIC ${CMAKE_CURRENT_LIST_DIR}/shim)
target_include_directories(eden_shader PUBLIC ${TP}/eden/src)
file(GLOB BOOST_INCLUDES ${TP}/boost/libs/*/include)
target_include_directories(eden_shader SYSTEM PUBLIC ${BOOST_INCLUDES})
target_link_libraries(eden_shader PUBLIC fmt::fmt sirit::sirit)
target_compile_options(eden_shader PRIVATE -fbracket-depth=1024 -w)

# Puente NVN -> recompilador (nuestro, GPL-3 por enlazar con Eden): lo que usan libnnsdk y shadertool.
target_sources(eden_shader PRIVATE ${CMAKE_CURRENT_LIST_DIR}/nvn_shader.cpp)
target_include_directories(eden_shader PUBLIC ${CMAKE_CURRENT_LIST_DIR})
