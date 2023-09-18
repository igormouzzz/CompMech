// ConsoleApplication1.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//


#include <immintrin.h>

#include <windows.h>
#include <ppl.h>
#include <omp.h>

#include <time.h>
#include <stdio.h>
#include <limits>
#include <algorithm>
#include <vector>
#include <ctime>
#include <iostream>
#include <numeric>

using namespace std;
using namespace concurrency;

int main()
{
    int NODES=100;
    double step1= 10000000;

    int countRDTSC = 100;
    unsigned long long t3 = __rdtsc();

    unsigned long long t1 = __rdtsc();

    for (int i = 0; i < countRDTSC; i++) { step1 *=i; }

    unsigned long long t2 = __rdtsc();
    std::cout <<"delta "<<t1-t3<< " res " << (t2 - t1)/(double) countRDTSC << std::endl<<"Count "<< step1/countRDTSC<<std::endl;


    char timestamp[100];
    struct timespec ts;
#define _mm256_mul_t _mm256_mul_pd
#define type double
#define __m256_t __m256d

    int step = 256 / 8 / sizeof(double);
    double k = 0.2, q = 0.3;
    double* K = &k;
    double* Q = &q;

    double dt;
    std::vector<double> A(NODES);
    std::vector<double> B(NODES);
    std::vector<double> C(NODES);
    std::vector<double> F(NODES);
    std::vector<double> D(NODES);

    std::iota(A.begin(), A.end(), 1);
    std::iota(B.begin(), B.end(), 2);
    std::iota(F.begin(), F.end(), -1);

    double* a = A.data(), * b = B.data(), * f = F.data(),*d=D.data();
    double* c = C.data();
    int r;
    double T = 0;
    timespec_get(&ts, TIME_UTC);
    dt = ts.tv_sec + static_cast<double>(ts.tv_nsec) / 1e9;
    t1 = __rdtsc();
    copy(A.begin(), A.end(), C.begin());
    t2 = __rdtsc();
    std::cout << " res_copy " << (t2 - t1) / (double)NODES << std::endl;

    timespec_get(&ts, TIME_UTC);
    std::cout << " copy " << ((ts.tv_sec + static_cast<double>(ts.tv_nsec) / 1e9) - dt) << " s" << std::endl;
//parallel_for 5.28607 s
    //#pragma omp parallel for
    for (int t = 0; t < 100; t++) {
        timespec_get(&ts, TIME_UTC);
        dt = ts.tv_sec + static_cast<double>(ts.tv_nsec) / 1e9;
        t1 = __rdtsc();

        {
            #pragma omp parallel for
            for (int i = 0; i < NODES; i++) {
                    d[i] = a[i] + b[i];
            }
        }
        t2 = __rdtsc();
        timespec_get(&ts, TIME_UTC);
        T += t2-t1;

    }
    std::cout << " omp for " << T / 100 /NODES << " c" << std::endl;
    for (int i = 0; i < NODES; i ++) {
        if (c[i] != d[i]) {
            //cout << i << ") " << c[i] << " != " << d[i] << " " << c[i] - d[i] << endl;
        }
    }
    t1 = __rdtsc();
    copy(A.begin(), A.end(), C.begin());
    t2 = __rdtsc();
    std::cout << " res_copy " << (t2 - t1) / (double)NODES << std::endl;

    T = 0;
    for (int t = 0; t < 100; t++) {
        timespec_get(&ts, TIME_UTC);
        dt = ts.tv_sec + static_cast<double>(ts.tv_nsec) / 1e9;
        t1 = __rdtsc();
        parallel_for(1, NODES, [&](int i)
             {
                    d[i] = a[i] + b[i];   
         });
        t2 = __rdtsc();
        timespec_get(&ts, TIME_UTC);
        T += t2 - t1;
        //T += (ts.tv_sec + static_cast<double>(ts.tv_nsec) / 1e9) - dt;
    }
    t1 = __rdtsc();
    copy(A.begin(), A.end(), C.begin());
    t2 = __rdtsc();
    std::cout << " res_copy " << (t2 - t1) / (double)NODES << std::endl;

    std::cout << " parallel_for " << T / 100/NODES << " s" << std::endl;
    for (int i = 0; i < NODES; i ++) {
        if (c[i] != d[i]) {
           // cout << i << ") " << c[i] << " != " << d[i] << " " << c[i] - d[i] << endl;
        }
    }
    t1 = __rdtsc();
    copy(A.begin(), A.end(), C.begin());
    t2 = __rdtsc();
    std::cout << " res_copy " << (t2 - t1) / (double)NODES << std::endl;

    T = 0;
    for (int t = 0; t < 10; t++) {

        timespec_get(&ts, TIME_UTC);
        dt = ts.tv_sec + static_cast<double>(ts.tv_nsec) / 1e9;

        //parallel_for(1, NODES/ step, [&](unsigned int i)
        t1 = __rdtsc();
        for (int i = 0; i < NODES/4; i++) 
            {
                   *(__m256_t*)(d + i * 4) = _mm256_mul_t(*(__m256_t*)(a + i * 4), *(__m256_t*)(b + i * 4));
            }
        timespec_get(&ts, TIME_UTC);
        t2 = __rdtsc();
        T += t2 - t1;

    }

    std::cout <<" "<<step << " intr " << T / 10 / NODES<< " s" << std::endl;

    T = 0;
    for (int t = 0; t < 100; t++) {
        timespec_get(&ts, TIME_UTC);
        dt = ts.tv_sec + static_cast<double>(ts.tv_nsec) / 1e9;
        t1 = __rdtsc();
        for (int i = 0; i < NODES; i++) {
                d[i] = a[i] + b[i];
        }
        t2 = __rdtsc();
        timespec_get(&ts, TIME_UTC);
        T += t2 - t1;
        //T += (ts.tv_sec + static_cast<double>(ts.tv_nsec) / 1e9) - dt;

    }
    std::cout << " for " << T / 100/NODES << " s" << std::endl;
    T = 0;
    return 0;
}



// Запуск программы: CTRL+F5 или меню "Отладка" > "Запуск без отладки"
// Отладка программы: F5 или меню "Отладка" > "Запустить отладку"

// Советы по началу работы 
//   1. В окне обозревателя решений можно добавлять файлы и управлять ими.
//   2. В окне Team Explorer можно подключиться к системе управления версиями.
//   3. В окне "Выходные данные" можно просматривать выходные данные сборки и другие сообщения.
//   4. В окне "Список ошибок" можно просматривать ошибки.
//   5. Последовательно выберите пункты меню "Проект" > "Добавить новый элемент", чтобы создать файлы кода, или "Проект" > "Добавить существующий элемент", чтобы добавить в проект существующие файлы кода.
//   6. Чтобы снова открыть этот проект позже, выберите пункты меню "Файл" > "Открыть" > "Проект" и выберите SLN-файл.
