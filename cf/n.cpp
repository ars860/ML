#include <iostream>
#include <vector>
#include <iomanip>
#include <unordered_map>
#include <map>
#include <set>
#include <algorithm>

using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    cout << setprecision(20);

    long long whyDoIEvenNeedThisShit;

    cin >> whyDoIEvenNeedThisShit;

    long long n;

    cin >> n;

    vector<pair<long long, long long>> features(n);

    unordered_map<long long, vector<long long>> y2xs;

    vector<long long> x(n);

    for (long long i = 0; i < n; i++) {
        cin >> features[i].first >> features[i].second;

        y2xs[features[i].second].push_back(features[i].first);
        x[i] = features[i].first;
    }

    sort(x.begin(), x.end());

    map<long long, pair<long long, long long>> x2num;
    {
        for (long long i = 0; i < x.size(); i++) {
            if (x2num.find(x[i]) == x2num.end()) {
                x2num[x[i]] = {-1, 0};
            }

            auto &val = x2num[x[i]];
            val.second++;
            if (val.first == -1) {
                val.first = i;
            }
        }
    }

    long long total = 0;
    for (long long i = 0; i < n; i++) {
        total = total +
                2 * x2num[features[i].first].first * features[i].first -
                2 * (n - (x2num[features[i].first].first + x2num[features[i].first].second)) * features[i].first;
    }

    long long total_test = 0;
    for (long long i = 0; i < n; i++) {
        for (long long j = 0; j < n; j++) {
            total_test += abs(features[i].first - features[j].first);
        }
    }

    long long inner_test = 0;
    long long inner = 0;
    for (auto &y2x : y2xs) {
        vector<long long> x(y2x.second.begin(), y2x.second.end());
        sort(x.begin(), x.end());

        map<long long, pair<long long, long long>> x2num;
        {
            for (long long i = 0; i < x.size(); i++) {
                if (x2num.find(x[i]) == x2num.end()) {
                    x2num[x[i]] = {-1, 0};
                }

                auto &val = x2num[x[i]];
                val.second++;
                if (val.first == -1) {
                    val.first = i;
                }
            }
        }

        for (long long i = 0; i < y2x.second.size(); i++) {
            inner = inner +
                    2 * x2num[y2x.second[i]].first * y2x.second[i] -
                    2 * (y2x.second.size() - (x2num[y2x.second[i]].first + x2num[y2x.second[i]].second)) * y2x.second[i];
        }

        for (long long x1 : y2x.second) {
            for (long long x2 : y2x.second) {
                inner_test += abs(x1 - x2);
            }
        }
    }

    cout << inner << '\n' << total - inner;
}