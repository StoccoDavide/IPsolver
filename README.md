# IPsolver

`IPsolver` is a header-only library that provides a simple interface for solving these types of optimization problems using an interior-point method. It is based on the [`Eigen`](https://eigen.tuxfamily.org/index.php?title=Main_Page) library and written in C++17.

Are you looking for the online documentation? Visit [this link](https://stoccodavide.github.io/IPsolver/)!

## Installation

### Quick and dirty

`IPsolver` is a header-only library that depends only on [`Eigen`](https://eigen.tuxfamily.org/index.php?title=Main_Page) (version >= 3.4.0), so the quick and dirty way of installing it is by simply copying the `include` directory to your project and make sure to have [`Eigen`](https://eigen.tuxfamily.org/index.php?title=Main_Page) available however you see fit. Alternatively, you can do things properly and use `CMake` (version >= 3.14).

### CMake

If you are using CMake, you can add the library as a subdirectory in your project.

```cmake
add_subdirectory(path/to/IPsolver)
target_link_libraries(your_target PRIVATE IPsolver::IPsolver)
```

You can use `FetchContent` to download the library from GitHub.

```cmake
include(FetchContent)

# Optionally specify a custom path to fetch content to
set(FETCHCONTENT_BASE_DIR "path/to/your/dependencies")
fetchcontent_declare(
  IPsolver
  GIT_REPOSITORY https://github.com/StoccoDavide/IPsolver.git
  GIT_TAG        main
)
fetchcontent_makeavailable(IPsolver)
target_link_libraries(your_target PRIVATE IPsolver::IPsolver)
```

If you already have `IPsolver` somewhere on your system, you can use `find_package` directly.

```cmake
# Optionally specify a custom path to find content from
list(APPEND CMAKE_PREFIX_PATH "path/to/your/dependencies")
find_package(
  IPsolver
  ${YOUR_DESIRED_IPsolver_VERSION}
  NO_MODULE
)

target_link_libraries(your_target PRIVATE IPsolver::IPsolver)
```

Since we are nice people, we also show you how to conditionally use `FetchContent` based if you already have the library or not.

```cmake
# Optionally specify a custom path to find content from
list(APPEND CMAKE_PREFIX_PATH "path/to/your/dependencies")
find_package(
  IPsolver
  ${YOUR_DESIRED_IPsolver_VERSION}
  NO_MODULE
)

if(NOT TARGET IPsolver::IPsolver)
  include(FetchContent)

  # Optionally specify a custom path to fetch content to
  set(FETCHCONTENT_BASE_DIR "path/to/your/dependencies")
  fetchcontent_declare(
    IPsolver
    GIT_REPOSITORY https://github.com/StoccoDavide/IPsolver.git
    GIT_TAG        main
  )

  fetchcontent_makeavailable(IPsolver)
endif()

target_link_libraries(your_target PRIVATE IPsolver::IPsolver)
```

## Authors

- Davide Stocco <br>
  University of Trento <br>
  Department of Industrial Engineering <br>
  email: davide.stocco@unitn.it

- Enrico Bertolazzi <br>
  University of Trento <br>
  Department of Industrial Engineering <br>
  email: enrico.bertolazzi@unitn.it

Aka...

```
▗▄▄▄  ▄   ▄  ▐▌    ▗▞▀▜▌▄▄▄▄     ▐▌    ▗▄▄▖ ▗▞▀▚▖ ▄▄▄ ▄   ▄
▐▌  █ █   █  ▐▌    ▝▚▄▟▌█   █    ▐▌    ▐▌ ▐▌▐▛▀▀▘█    █   █
▐▌  █  ▀▄▀▗▞▀▜▌         █   █ ▗▞▀▜▌    ▐▛▀▚▖▝▚▄▄▖█     ▀▀▀█
▐▙▄▄▀     ▝▚▄▟▌               ▝▚▄▟▌    ▐▙▄▞▘          ▄   █
                                                       ▀▀▀
```

## License

The `IPsolver` project is distributed under the BSD 2-Clause License - see the [LICENSE](https://StoccoDavide.github.io/IPsolver/LICENSE) file for details.

Here's what the license entails:

1. Anyone can copy, modify and distribute this software.
2. You have to include the license and copyright notice with each and every distribution.
3. You can use this software privately.
4. You can use this software for commercial purposes.
5. This software is provided without warranty.
6. The software author or license can not be held liable for any damages inflicted by the software.
