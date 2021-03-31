#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <cassert>

#define _USE_MATH_DEFINES

#include <cmath>
#include <functional>
#include <numeric>
//#include "../pybind/pybind11/include/pybind11/pybind11.h"

//namespace py = pybind11;

using namespace std;

enum class Window {
    fixed, variable
};

double manhattanDistance(vector<double> &x, vector<double> &y) {
    assert(x.size() == y.size());
    int n = x.size();
    double dist = 0;
    for (int i = 0; i < n; i++) {
        dist += abs(x[i] - y[i]);
    }
    return dist;
}

double euclideanDistance(vector<double> &x, vector<double> &y) {
    assert(x.size() == y.size());
    int n = x.size();
    double dist = 0;
    for (int i = 0; i < n; i++) {
        dist += (x[i] - y[i]) * (x[i] - y[i]);
    }
    return sqrt(dist);
}

double chebyshevDistance(vector<double> &x, vector<double> &y) {
    assert(x.size() == y.size());
    int n = x.size();
    double dist = 0;
    for (int i = 0; i < n; i++) {
        dist = max(dist, abs(x[i] - y[i]));
    }
    return dist;
}

typedef double (*MetricF)(vector<double> &, vector<double> &);

MetricF parseDist() {
    string s;
    cin >> s;
    if (s == "manhattan")
        return manhattanDistance;
    if (s == "euclidean")
        return euclideanDistance;
    if (s == "chebyshev")
        return chebyshevDistance;
    throw runtime_error("Unknown distance! " + s);
}

double uniformKernel(double x) {
    if (abs(x) >= 1.0) {
        return 0.0;
    }
    return 0.5;
}

double triangularKernel(double x) {
    if (abs(x) >= 1.0) {
        return 0.0;
    }
    return 1.0 - abs(x);
}

double epanechnikovKernel(double x) {
    if (abs(x) >= 1.0) {
        return 0.0;
    }
    return 0.75 * (1.0 - x * x);
}

double quarticKernel(double x) {
    if (abs(x) >= 1.0) {
        return 0.0;
    }
    return 15.0 / 16.0 * (1.0 - x * x) * (1.0 - x * x);
}

double triweightKernel(double x) {
    if (abs(x) >= 1.0) {
        return 0.0;
    }
    return 35.0 / 32.0 * (1.0 - x * x) * (1.0 - x * x) * (1.0 - x * x);
}

double tricubeKernel(double x) {
    if (abs(x) >= 1.0) {
        return 0.0;
    }
    return 70.0 / 81.0 * (1.0 - abs(x * x * x)) * (1.0 - abs(x * x * x)) * (1.0 - abs(x * x * x));
}

double gaussianKernel(double x) {
    return 1.0 / sqrt(2.0 * M_PI) * pow(M_E, -0.5 * x * x);
}

double cosineKernel(double x) {
    if (abs(x) >= 1.0) {
        return 0.0;
    }
    return M_PI * 0.25 * cos(M_PI * 0.5 * x);
}

double logisticKernel(double x) {
    return 1.0 / (pow(M_E, x) + 2.0 + pow(M_E, -x));
}

double sigmoidKernel(double x) {
    return 2.0 / (M_PI * (pow(M_E, x) + pow(M_E, -x)));
}

typedef double (*KernelF)(double);

KernelF parseKernel() {
    string s;
    cin >> s;
    if (s == "uniform")
        return uniformKernel;
    if (s == "triangular")
        return triangularKernel;
    if (s == "epanechnikov")
        return epanechnikovKernel;
    if (s == "quartic")
        return quarticKernel;
    if (s == "triweight")
        return triweightKernel;
    if (s == "tricube")
        return tricubeKernel;
    if (s == "gaussian")
        return gaussianKernel;
    if (s == "cosine")
        return cosineKernel;
    if (s == "logistic")
        return logisticKernel;
    if (s == "sigmoid")
        return sigmoidKernel;
    throw runtime_error("Unknown kernel!");
}

bool isFinite(KernelF kernel) {
    return !(kernel == gaussianKernel || kernel == logisticKernel || kernel == sigmoidKernel);
}

pair<Window, double> parseWindow() {
    string s;
    double parameter;
    cin >> s >> parameter;
    if (s == "fixed")
        return {Window::fixed, parameter};
    if (s == "variable")
        return {Window::variable, parameter};
    throw runtime_error("Unknown window!");
}

int precision = 8;

double regression(vector<vector<double>>& d, const vector<double>& classes, vector<double>& request) {
    int n = d.size();

    MetricF dist = parseDist();
    KernelF kernel = parseKernel();
    pair<Window, double> window = parseWindow();

    vector<double> dists(n);

    for (int i = 0; i < n; i++) {
        dists[i] = dist(request, d[i]);
    }

    vector<pair<double, int>> distsSorted(n);
    for (int i = 0; i < n; i++) {
        distsSorted[i] = {dists[i], i};
    }
    sort(distsSorted.begin(), distsSorted.end());

    if (window.first == Window::variable) {
        window.second = window.second >= n ? throw runtime_error("never happens") : distsSorted[window.second].first;
    }

    // check window
    if (window.second <= distsSorted[0].first) {
        if (window.second == distsSorted[0].first && distsSorted[0].first == 0) {
            double res = 0;
            int i = 0;
            while (distsSorted[i].first == 0 && i < n) {
                res += classes[distsSorted[i].second];
                i++;
            }

            return res / i;
        }

        return accumulate(classes.begin(), classes.end(), 0.0) / classes.size();
    }

    double up = 0, down = 0;
    for (int i = 0; i < n; i++) {
        up += classes[i] * kernel(dists[i] / window.second);
        down += kernel(dists[i] / window.second);
    }

    return up / down;
}

//PYBIND11_MODULE(regressionCalculator, m) {
//    m.doc() = "pybind11 example plugin"; // optional module docstring
//
//    m.def("add", &regression, "A function which adds two numbers");
//}

int main() {
    int n, m;

    cin >> n >> m;

    vector<vector<double>> d(n, vector<double>(m));
    vector<double> classes(n);
    vector<double> request(m);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            cin >> d[i][j];
        }
        cin >> classes[i];
    }

    for (int i = 0; i < m; i++) {
        cin >> request[i];
    }

    MetricF dist = parseDist();
    KernelF kernel = parseKernel();
    pair<Window, double> window = parseWindow();

    vector<double> dists(n);

    for (int i = 0; i < n; i++) {
        dists[i] = dist(request, d[i]);
    }

    vector<pair<double, int>> distsSorted(n);
    for (int i = 0; i < n; i++) {
        distsSorted[i] = {dists[i], i};
    }
    sort(distsSorted.begin(), distsSorted.end());

    if (window.first == Window::variable) {
        window.second = window.second >= n ? throw runtime_error("never happens") : distsSorted[window.second].first;
    }

    // check window
    if (window.second <= distsSorted[0].first) {
        if (window.second == distsSorted[0].first && distsSorted[0].first == 0) {
            double res = 0;
            int i = 0;
            while (distsSorted[i].first == 0 && i < n) {
                res += classes[distsSorted[i].second];
                i++;
            }
            cout << setprecision(precision) << res / i;
            return 0;
        }
        cout << setprecision(precision) << accumulate(classes.begin(), classes.end(), 0.0) / classes.size();
        return 0;
    }

    double up = 0, down = 0;
    for (int i = 0; i < n; i++) {
        up += classes[i] * kernel(dists[i] / window.second);
        down += kernel(dists[i] / window.second);
    }

    cout << setprecision(precision) << up / down;
}
