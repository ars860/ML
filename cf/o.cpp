#include <iostream>
#include <vector>
#include <map>
#include <iomanip>

using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    cout << setprecision(20);

    int maxx;
    cin >> maxx;

    int n;
    cin >> n;

    vector<int> x(n);
    vector<int> y(n);

    map<int, map<int, int>> x2y;
    map<int, int> x2sz;

    for (int i = 0; i < n; i++) {
        cin >> x[i] >> y[i];

        x2y[x[i]][y[i]]++;
        x2sz[x[i]]++;
    }

    int x_amount = x2y.size();

    vector<long double> eyx2(x_amount);

    int index = 0;
    for (auto &xy : x2y) {
        for (auto &y2cnt : xy.second) {
            eyx2[index] += y2cnt.second / (long double) x2sz[xy.first] * y2cnt.first;
        }
        index++;
    }

    for (long double &i : eyx2) {
        i = i * i;
    }

    vector<long double> ey2x(x_amount);

    index = 0;
    for (auto &xy : x2y) {
        for (auto &y2cnt : xy.second) {
            ey2x[index] += y2cnt.second / (long double) x2sz[xy.first] * y2cnt.first * y2cnt.first;
        }
        index++;
    }

    vector<long double> sub(x_amount);
    for (int i = 0; i < x_amount; i++) {
        sub[i] = ey2x[i] - eyx2[i];
    }

    index = 0;
    for (auto &xy : x2y) {
        sub[index] *= x2sz[xy.first] / (long double) n;
        index++;
    }

    long double res = 0;
    for (long double v : sub) {
        res += v;
    }

    cout << res;

//    int tet = 123 + 123;
}