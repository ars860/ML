#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <set>
#include <map>

using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    cout << setprecision(20);

    int n;

    cin >> n;

    vector<pair<int, int>> features(n);
    long double mean_x = 0, mean_y = 0;

    set<int> x;
    set<int> y;

    for (int i = 0; i < n; i++) {
        cin >> features[i].first >> features[i].second;

        x.insert(features[i].first);
        y.insert(features[i].second);
    }


    map<int, int> x2num;
    {
        vector<int> x_vec(x.begin(), x.end());
        for (int i = 0; i < x_vec.size(); i++) {
            x2num[x_vec[i]] = i;
        }
    }

    map<int, int> y2num;
    {
        vector<int> y_vec(y.begin(), y.end());
        for (int i = 0; i < y_vec.size(); i++) {
            y2num[y_vec[i]] = i;
        }
    }

    mean_x = x2num.size() / 2;
    mean_y = y2num.size() / 2;

    long double top = 0, botl = 0, botr = 0;
    for (int i = 0; i < n; i++) {
        top += (x2num[features[i].first] - mean_x) * (y2num[features[i].second] - mean_y);
        botl += (x2num[features[i].first] - mean_x) * (x2num[features[i].first] - mean_x);
        botr += (y2num[features[i].second] - mean_y) * (y2num[features[i].second] - mean_y);
    }

    cout << (botl * botr == 0 ? 0 : top / sqrt(botl * botr));
}