# miniAero CFD Mini-Application

> [!CAUTION]
> Unofficial fork of miniFE by with some tooling updates

* CMake support
  * inlucding CTest
* Github Actions CI tests
* Kokkos 4.3
  * `Device::fence()` -> `Device().fence()`
  * `Kokkos::Impl::Timer` -> `Kokkos::Timer`
  * `View::dimension_0` -> `View::extent(0)`
* Testing scripts updated for Python 3
* C++17 support
  * `std::random_shuffle` -> `std::shuffle`

[Click here](https://github.com/Mantevo/miniAero/compare/master..cwpearson:miniAero:master) to see all comparisons in one place

> [!NOTE]  
> Original Readme follows verbatim

# miniAero CFD Mini-Application

MiniAero is a mini-application for the evaulation of programming models and hardware for next generation platforms. MiniAero is an explicit (using RK4) unstructured finite volume code that solves the compressible Navier-Stokes equations. Both inviscid and viscous terms are included. The viscous terms can be optionally included or excluded.

For more details, please see the following reference:

K.J. Franko, T.C. Fisher, P.T. Lin, and S.W. Bova, CFD for next generation hardware: experiences with proxy applications, AIAA 2015-3053, 22nd AIAA Computational Fluid Dynamics Conference, Dallas TX, June 2015, DOI: doi.org/10.2514/6.2015-3053
