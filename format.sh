#!/bin/sh
find src tests -name "*.cpp" -o -name "*.h" | xargs clang-format -i
