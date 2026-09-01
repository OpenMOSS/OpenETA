project(tetgen)

file(GLOB_RECURSE TETGEN_SOURCES "tetgen/*.cxx" "tetgen/*.h")

add_library(tetgen STATIC ${TETGEN_SOURCES})
target_compile_definitions(tetgen PRIVATE -DTETLIBRARY)
# Generate position independent code
set_target_properties(tetgen PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(tetgen PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})


