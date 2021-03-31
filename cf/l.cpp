#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    cout << setprecision(20);

    int n;

    cin >> n;

    vector<pair<long double, long double>> features(n);
    long double mean_x = 0, mean_y = 0;

    for (int i = 0; i < n; i++) {
        cin >> features[i].first >> features[i].second;
        mean_x += features[i].first;
        mean_y += features[i].second;
    }

    mean_x /= features.size();
    mean_y /= features.size();

    long double top = 0, botl = 0, botr = 0;
    for (int i = 0; i < n; i++) {
        top += (features[i].first - mean_x) * (features[i].second - mean_y);
        botl += (features[i].first - mean_x) * (features[i].first - mean_x);
        botr += (features[i].second - mean_y) * (features[i].second - mean_y);
    }

    cout << (botl * botr == 0 ? 0 : top / sqrt(botl * botr));
}