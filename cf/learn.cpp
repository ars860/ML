#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <cassert>

#define _USE_MATH_DEFINES

#include <cmath>
#include <functional>
#include <numeric>

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

bool isFinite(KernelF kernel) {
    return !(kernel == gaussianKernel || kernel == logisticKernel || kernel == sigmoidKernel);
}

int precision = 8;

void addIn(vector<double> &x, const vector<double> &y) {
    assert(x.size() == y.size());
    for (int i = 0; i < x.size(); i++) {
        x[i] += y[i];
    }
}

vector<double> accumulateVecs(const vector<vector<double>> &vectors) {
    vector<double> result(vectors[0].size());
    for (const vector<double> &vec : vectors) {
        addIn(result, vec);
    }
    return result;
}

vector<double> add(vector<double> &x, vector<double> &y) {
    vector<double> ans(x);
    addIn(ans, y);

    return ans;
}

vector<double> mulConst(const vector<double> &x, double y) {
    vector<double> res(x);
    for (double &val : res) {
        val *= y;
    }

    return res;
}

vector<double> &divConstIn(vector<double> &&x, double y) {
    for (double &val : x) {
        val /= y;
    }

    return x;
}

vector<double>
regression(vector<reference_wrapper<vector<double>>> &d, const vector<vector<double>> &classes, vector<double> &request,
           MetricF dist, KernelF kernel, pair<Window, double> &&window) {
    int n = d.size();
    int classSize = classes[0].size();

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
            vector<double> res(classSize);
            int i = 0;
            while (distsSorted[i].first == 0 && i < n) {
                addIn(res, classes[distsSorted[i].second]);
                i++;
            }

            return divConstIn(move(res), i);
        }

        return divConstIn(accumulateVecs(classes), classes.size());
    }

    vector<double> up(classSize);
    double down = 0;
    for (int i = 0; i < n; i++) {
        addIn(up, mulConst(classes[i], kernel(dists[i] / window.second)));
        down += kernel(dists[i] / window.second);
    }

    return divConstIn(move(up), down);
}

string DIVIDER = "<-------------------------------------------------------------->\n";

pair<double, double> calcFScores(vector<vector<int>> &confusionMatrix);

string windowToString(Window window) {
    return (window == Window::fixed ? "fixed" : "variable");
}

string kernelToString(KernelF kernel) {
    if (kernel == uniformKernel)
        return "uniform";
    if (kernel == triangularKernel)
        return "triangular";
    if (kernel == epanechnikovKernel)
        return "epanechnikov";
    if (kernel == quarticKernel)
        return "quartic";
    if (kernel == triweightKernel)
        return "triweight";
    if (kernel == tricubeKernel)
        return "tricube";
    if (kernel == gaussianKernel)
        return "gaussian";
    if (kernel == cosineKernel)
        return "cosine";
    if (kernel == logisticKernel)
        return "logistic";
    if (kernel == sigmoidKernel)
        return "sigmoid";
    throw runtime_error("Unknown kernel!");
}

string metricToString(MetricF &s) {
    if (s == manhattanDistance)
        return "manhattan";
    if (s == euclideanDistance)
        return "euclidean";
    if (s == chebyshevDistance)
        return "chebyshev";
    throw runtime_error("Unknown metric!");
}

struct Result {
public:
    Result(MetricF metric, KernelF kernel, pair<Window, double> window, vector<vector<int>> &confusionMatrix,
           bool isOneHot) : metric(
            metric), kernel(kernel), window(std::move(window)), isOneHot(isOneHot) {
        auto fScores = calcFScores(confusionMatrix);
        macroFScore = fScores.first;
        microFScore = fScores.second;
    }

    string csv() {
        std::stringstream stream;
        stream << metricToString(metric) << "," << kernelToString(kernel) << "," << windowToString(window.first) << ","
               << to_string(window.second) << "," <<
               std::fixed << std::setprecision(8) << macroFScore << "," << microFScore << ","
               << (isOneHot ? "true" : "false");

        return stream.str();
    }

    static string header() {
        return R"("metric","kernel","window_type","window_size","macroFScore","microFScore","isOneHot")";
    }

private:
    MetricF metric;
    KernelF kernel;
    pair<Window, double> window;
    double macroFScore;
    double microFScore;
    bool isOneHot;
};

void printResCsvStyle(vector<Result> &results) {
    for (Result &result : results) {
        cout << result.csv() << "\n";
    }
}

int DEFAULT_STEPS_AMOUNT = 20;

pair<double, double> calcFScores(vector<vector<int>> &confusionMatrix) {
    int k = confusionMatrix.size();

    vector<double> p(k);
    vector<double> c(k);
    vector<double> t(k);
    double all = 0;

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            double cmij = confusionMatrix[i][j];

            all += cmij;
            p[j] += cmij;
            c[i] += cmij;
            if (i == j) {
                t[i] = cmij;
            }
        }
    }

    double weightedF1 = 0;

    for (int i = 0; i < k; i++) {
        double prec = t[i] / p[i],
                recoll = t[i] / c[i],
                f1 = 2 * prec * recoll / (prec + recoll);
        if (p[i] != 0 && c[i] != 0 && prec + recoll != 0)
            weightedF1 += f1 * c[i];
    }

    weightedF1 /= all;

    double precw = 0;
    double recallw = 0;

    for (int i = 0; i < k; i++) {
        if (p[i] != 0) {
            precw += t[i] * c[i] / p[i];
            recallw += t[i];
        }
    }
    precw /= all;
    recallw /= all;

    return {2 * precw * recallw / (precw + recallw), weightedF1};
}

int getClassFromOneHot(vector<double> &&oneHot, bool isOneHot) {
    if (!isOneHot) {
        return (int)round(oneHot[0]);
    }

    int maximumIndex = 0;
    for (int i = 0; i < oneHot.size(); i++) {
        if (oneHot[i] > oneHot[maximumIndex]) {
            maximumIndex = i;
        }
    }

    return maximumIndex + 1;
}

int main() {
    int n, m, stepsAmount;

    cin >> n >> m;

    vector<vector<double>> d(n, vector<double>(m));
    vector<double> classes(n);

    int classesAmount = 0;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            cin >> d[i][j];
        }
        cin >> classes[i];
        classesAmount = max(classesAmount, (int) classes[i]);
    }

    cin >> stepsAmount;
    if (stepsAmount == -1) {
        stepsAmount = DEFAULT_STEPS_AMOUNT;
    }

    cout << Result::header() << "\n";
    vector<Result> results;
    for (bool isOneHot : {false, true}) {
        vector<vector<double>> classesVectorized;
        for (double clazz : classes) {
            if (isOneHot) {
                vector<double> oneHotted(classesAmount);
                oneHotted[clazz - 1] = 1;
                classesVectorized.push_back(oneHotted);
            } else {
                classesVectorized.emplace_back(1, clazz);
            }
        }

        for (MetricF metric : {manhattanDistance, euclideanDistance, chebyshevDistance}) {
            double maximumPossibleDistance = 0;

            for (vector<double> &row1 : d) {
                for (vector<double> &row2 : d) {
                    maximumPossibleDistance = max(maximumPossibleDistance, metric(row1, row2));
                }
            }

            for (KernelF kernel : {uniformKernel, triangularKernel, epanechnikovKernel, quarticKernel, triweightKernel,
                                   tricubeKernel, gaussianKernel,
                                   cosineKernel, logisticKernel, sigmoidKernel}) {
                for (Window windowType : {Window::variable, Window::fixed}) {
                    for (int windowStep = 1; windowStep < stepsAmount; windowStep++) {
                        double windowSize = windowStep *
                                            (windowType == Window::fixed ? (maximumPossibleDistance / stepsAmount)
                                                                            : (d.size() - 2.0) / stepsAmount);
                        vector<vector<int>> confusionMatrix(classesAmount, vector<int>(classesAmount));

                        for (int leftOutId = 0; leftOutId < d.size(); leftOutId++) {
                            vector<reference_wrapper<vector<double>>> withoutLeftOut;
                            for (int i = 0; i < d.size(); i++) {
                                if (i != leftOutId) {
                                    withoutLeftOut.emplace_back(d[i]);
                                }
                            }
                            vector<double> &leftOut = d[leftOutId];

                            int actualClass = (int) classes[leftOutId];
                            int predictedClass =
                                    getClassFromOneHot(
                                            regression(withoutLeftOut, classesVectorized, leftOut, metric, kernel,
                                                       {windowType, windowSize}), isOneHot);

                            confusionMatrix[actualClass - 1][predictedClass - 1]++;
                        }

                        results.push_back(Result(metric, kernel, {windowType, windowSize}, confusionMatrix, isOneHot));
                    }
                }
            }
        }
    }

    printResCsvStyle(results);
}
