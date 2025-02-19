add_library(AnyMovable AnyMovable.cpp AnyMovable.h)

target_include_directories(AnyMovable INTERFACE "${PROJECT_SOURCE_DIR}/pkg/AnyObject")