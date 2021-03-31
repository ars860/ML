cmake_minimum_required(VERSION 3.17)
project(cf)

set(CMAKE_CXX_STANDARD 14)

#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -gdwarf-3")
#set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -gdwarf-3")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++ -static-libgcc -Wall")

add_executable(a a.cpp)
add_executable(b b.cpp)
add_executable(c c.cpp)
add_executable(learn learn.cpp)

add_executable(d d.cpp)
add_executable(4_for_lab 4_for_lab.cpp)

add_executable(5 e.cpp)
add_executable(svm_lab svm_lab.cpp)

add_executable(6 f.cpp)
add_executable(bayes_lab_LAMBDAS bayes_lab.cpp)

add_executable(g g.cpp)

add_executable(h h.cpp)

add_executable(l l.cpp)

add_executable(m m.cpp)

add_executable(n n.cpp)

add_executable(o o.cpp)

add_executable(q q.cpp)

add_executable(p p.cpp)

add_executable(i I.cpp)