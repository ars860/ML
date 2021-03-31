#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int main() {
    int n, m, k;

    cin >> n >> m >> k;
    vector<pair<int, int>> classes(n);

    for (int i = 0; i < n; i++) {
        int curClass;
        cin >> curClass;
        classes[i] = {curClass, i + 1};
    }

    vector<vector<int>> groups(k, vector<int>());

    sort(classes.begin(), classes.end());

    for (int i = 0; i < classes.size(); i++) {
        groups[i % k].push_back(classes[i].second);
    }


    for (vector<int> &group : groups) {
        cout << group.size() << " ";
        for (int element : group) {
            cout << element << " ";
        }
        cout << '\n';
    }
}
