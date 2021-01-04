

set(utils_source_files
 filesystem.cpp
 timing.cpp
)

include_directories(.)

add_library(utils STATIC 
  ${utils_source_files} 
)

SET_TARGET_PROPERTIES( utils PROPERTIES COMPILE_FLAGS -fPIC)
