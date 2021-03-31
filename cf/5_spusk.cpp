#include <iostream>
#include <vector>
#include <cassert>
#include <ctime>
#include <iomanip>
#include <set>
#include <map>

using namespace std;

void printVec(vector<double> &x) {
    for (double val : x) {
        cout << val << " ";
    }
    cout << '\n';
}

double scalar(vector<double> &x, vector<double> &y) {
    assert(x.size() == y.size());
    double res = 0;
    for (int i = 0; i < x.size(); i++) {
        res += x[i] * y[i];
    }

    return res;
}


vector<double> sub(vector<double> &&x, vector<double> &&y) {
    assert(x.size() == y.size());
    vector<double> res(x.size());
    for (int i = 0; i < x.size(); i++) {
        res[i] = x[i] - y[i];
    }

    return res;
}

vector<double> add(vector<double> &&x, vector<double> &&y) {
    assert(x.size() == y.size());
    vector<double> res(x.size());
    for (int i = 0; i < x.size(); i++) {
        res[i] = x[i] + y[i];
    }

    return res;
}

vector<double> mulConst(vector<double> &x, double y) {
    vector<double> res(x.size());
    for (int i = 0; i < res.size(); i++) {
        res[i] = x[i] * y;
    }

    return res;
}

vector<double> mulConst(vector<double> &&x, double y) {
    return mulConst(x, y);
}

vector<double> sign(vector<double> &x) {
    vector<double> res(x.size());
    for (int i = 0; i < x.size(); i++) {
        res[i] = x[i] > 0 ? 1 : -1;
    }

    return res;
}

vector<double> gradientStohastic(vector<vector<double>> &d, vector<double> &a, vector<double> &y, int index) {
    int n = d.size(), m = a.size();

    vector<double> res(m, 0.0);

    for (int j = 0; j < m; j++) {
        res[j] = -1;
        for (int i = 0; i < n; i++) {
            if (i == j) {
                res[j] += a[i] * y[i] * y[j] * d[i][j];
            } else {
                res[j] += 0.5 * a[i] * y[i] * y[j] * d[i][j];
            }
        }
    }

    return res;
}

vector<double>
doShit(vector<vector<double>> &d, vector<double> &classes, double C,
       double ridge, double lasso,
       vector<vector<double>> &d_test, vector<double> &classes_test) {
    int n = d.size(), m = d[0].size() - 1;

    vector<double> w(m + 1);
    for (double &weight : w) {
        weight = ((double) rand() / RAND_MAX) * C;
    }

    int i = 0;
    while (i < 2000) {
        i++;
        double step = 1.0 / i;

        if (i % 10 == 0) {
            step = 0.1;
        }

        int randomObj = rand() % n;

        w = sub(
                sub(
                        mulConst(w, 1.0 - ridge * step),
                        mulConst(gradientStohastic(d, w, classes, randomObj), step)),
                mulConst(sign(w), lasso)
        );

    }

    return w;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    cout << setprecision(20);

    int n;
    cin >> n;
    vector<vector<double>> d(n, vector<double>(n));
    vector<double> y(n);
    vector<double> a(n, 0);
    vector<double> w(n, 0);
//    vector<double> a2(n, 0);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cin >> d[i][j];
        }
        cin >> y[i];
    }

    int upper_bound;
    cin >> upper_bound;


}