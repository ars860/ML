#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <iomanip>

using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    cout << setprecision(20);

    int k1, k2;

    cin >> k1 >> k2;

    int n;
    cin >> n;

    vector<int> x(n);
    vector<int> y(n);

    map<int, double> x2p;

    map<int, map<int, double>> x2y2cnt;

    for (int i = 0; i < n; i++) {
        cin >> x[i] >> y[i];

        x2y2cnt[x[i]][y[i]]++;
        x2p[x[i]]++;
    }

    double res = 0;

    for (auto& xycnt : x2y2cnt) {
        double tmp_sum = 0;
        double cnt = x2p[xycnt.first];

        for (auto& y2cnt : xycnt.second) {
            tmp_sum += y2cnt.second / cnt * log(y2cnt.second / cnt);
        }

        res -= cnt / n * tmp_sum;
    }

    cout << res;
}