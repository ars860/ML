#include <iostream>
#include <vector>
#include <cassert>
#include <iomanip>
#include <cmath>

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

//vector<double> add(vector<double> &x, vector<double> &&y) {
//    assert(x.size() == y.size());
//    vector<double> res(x.size());
//    for (int i = 0; i < x.size(); i++) {
//        res[i] = x[i] + y[i];
//    }
//
//    return res;
//}

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

vector<double> a;
vector<double> target;
vector<double> error_cache;
vector<vector<double>> d;
double C = 1;
double eps = 0.001;
double b = 0;
double tol = 0.001;

//void calcB() {
////    if (b != -1) {
////        return;
////    }
//
//    vector<double> w(d.size(), 0);
//
//    for (int i = 0; i < d.size(); i++) {
//        w = add(w, mulConst(mulConst(d[i], target[i]), a[i]));
//    }
//
//    int notZeroAIndex = 0;
//
//    for (int i = 0; i < a.size(); i++) {
//        if (a[i] > 0) {
//            notZeroAIndex = i;
//            continue;
//        }
//    }
//
////    assert(notZeroAIndex != -1);
//
//    b = scalar(w, d[notZeroAIndex]) - target[notZeroAIndex];
//}

double calcPrediction(int index) {
    assert(d.size() == d[0].size());
    assert(d.size() == a.size());
    assert(d.size() == target.size());

    double ans = 0;

//    calcB();
    for (int i = 0; i < d.size(); i++) {
        if (a[i] > 0) {
            ans += a[i] * target[i] * d[index][i];
        }
    }

    ans -= b;

    return ans;
}

int takeStep(int i1, int i2) {
    if (i1 == i2) {
        return 0;
    }

    double alph1 = a[i1];
    double alph2 = a[i2];
    double y1 = target[i1];
    double y2 = target[i2];
    double s = y1 * y2;

    double E1 = alph1 > 0 && alph1 < C ? error_cache[i1] : calcPrediction(i1) - y1;
    double E2 = alph2 > 0 && alph2 < C ? error_cache[i2] : calcPrediction(i2) - y2;

    double L, H;
    if (y1 == y2) {
        if (alph1 + alph2 > C) {
            L = alph1 + alph2 - C;
            H = C;
        } else {
            L = 0;
            H = alph1 + alph2;
        }
    } else {
        if (alph1 - alph2 > 0) {
            L = 0;
            H = C - (alph1 - alph2);
        } else {
            L = alph2 - alph1;
            H = C;
        }
    }

    if (L == H) {
        return 0;
    }

    double k11 = d[i1][i1];
    double k12 = d[i1][i2];
    double k22 = d[i2][i2];
    double eta = 2 * k12 - k11 - k22;

    double a2;
    if (eta < 0) {
        a2 = alph2 + y2 * (E2 - E1) / eta;
        if (a2 < L) {
            a2 = L;
        } else {
            if (a2 > H) {
                a2 = H;
            }
        }
    } else {
//        calcB();

        double c1 = eta / 2;
        double c2 = y2 * (E1 - E2) - eta * alph2;
        double Lobj = c1 * L * L + c2 * L;
        double Hobj = c1 * H * H + c2 * H;

        if (Lobj > Hobj + eps) {
            a2 = L;
        } else {
            if (Lobj < Hobj - eps) {
                a2 = H;
            } else {
                a2 = alph2;
            }
        }
    }

    if (abs(a2 - alph2) < eps * (a2 + alph2 + eps)) {
        return 0;
    }

    double a1 = alph1 + s * (alph2 - a2);

    if (a1 < 0) {
        a2 += s * a1;
        a1 = 0;
    } else if (a1 > C) {
        a2 += s * (a1 - C);
        a1 = C;
    }


//    Update threshold to reflect change in Lagrange multipliers
//    b = -1;

    double bnew;
    if (a1 > 0 && a1 < C) {
        bnew = b + E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12;
    } else {
        if (a2 > 0 && a2 < C) {
            bnew = b + E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22;
        } else {
            double b1 = b + E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12;
            double b2 = b + E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22;

            bnew = (b1 + b2) / 2;
        }
    }
    double delta_b = bnew - b;
    b = bnew;

//    Update weight vector to reflect change in a1 & a2, if SVM is linear ???

//    Update error cache using new Lagrange multipliers
    for (int i = 0; i < d.size(); i++) {
        if (0 < a[i] && a[i] < C) {
            error_cache[i] +=
                    y1 * (a1 - alph1) * d[i1][i] +
                    y2 * (a2 - alph2) * d[i2][i] -
                    delta_b;
        }
    }

    error_cache[i1] = 0;
    error_cache[i2] = 0;

    a[i1] = a1;
    a[i2] = a2;

    return 1;
}

int examineExample(int i1) {
    double y1 = target[i1];
    double alph1 = a[i1];

    double E1;
    if (alph1 > 0 && alph1 < C) {
        E1 = error_cache[i1];
    } else {
        E1 = calcPrediction(i1) - y1;
    }

    double r1 = y1 * E1;

    if ((r1 < -tol && alph1 < C) || (r1 > tol && alph1 > 0)) {
        pair<double, int> maxDiff = {-1, -1};

        for (int i = 0; i < a.size(); i++) {
            if (a[i] > 0 && a[i] < C) {
                maxDiff = max(maxDiff, {abs(E1 - error_cache[i]), i});
            }
        }

        if (maxDiff.second != -1) {
            if (takeStep(i1, maxDiff.second))
                return 1;
        }

        for (int i = 0; i < a.size(); i++) {
            if (a[i] > 0 && a[i] < C) {
                if (takeStep(i1, i))
                    return 1;
            }
        }

        for (int i = 0; i < a.size(); i++) {
            if (takeStep(i1, i)) {
                return 1;
            }
        }

    }

    return 0;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    cout << setprecision(20);

//    freopen("input.txt", "r", stdin);

    int n;
    cin >> n;

    d = vector<vector<double>>(n, vector<double>(n));
    target = vector<double>(n);
    a = vector<double>(n);

    error_cache = vector<double>(n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cin >> d[i][j];
        }
        cin >> target[i];
    }

    int up;
    cin >> up;

    int numChanged = 0;
    bool examineAll = true;

    while (numChanged > 0 || examineAll) {
        numChanged = 0;
        if (examineAll) {
            for (int i = 0; i < n; i++) {
                numChanged += examineExample(i);
            }
        } else {
            for (int i = 0; i < n; i++) {
                if (a[i] != 0 && a[i] != C) {
                    numChanged += examineExample(i);
                }
            }
        }

        if (examineAll) {
            examineAll = false;
        } else {
            if (numChanged == 0) {
                examineAll = true;
            }
        }
    }


    for (int i = 0; i < d.size(); i++) {
        cout << a[i] << " ";
    }
    cout << -b;
}


