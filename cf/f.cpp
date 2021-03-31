#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <cmath>
#include <iomanip>

using namespace std;

// (2 + 1) / (2 + 2 * alpha)

long double pxc(string const &x, int c, map<string, map<int, int>> &word2classes, map<int, int> &class2cnt, long double alpha) {
    long double countX = word2classes[x][c];
    long double countC = class2cnt[c];

    return (countX + alpha) / (countC + 2.0 * alpha);
}

long double py(int y, map<int, int> &class2cnt, int classesCnt) {
    return (long double) class2cnt[y] / classesCnt;
}


int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    cout << setprecision(20);

//    freopen("bayes.txt", "r", stdin);

    int k;
    cin >> k;

    vector<long double> lambda(k);
    for (int i = 0; i < k; i++) {
        cin >> lambda[i];
    }

    long double alpha;
    cin >> alpha;

    int n;
    cin >> n;

    vector<vector<string>> messages(n);
    vector<int> target(n);

    map<string, map<int, int>> word2classes;
    map<int, int> class2cnt;
    set<int> classes;

    set<string> dataset_words;

    for (int i = 0; i < n; i++) {
        cin >> target[i];

        classes.insert(target[i]);
        class2cnt[target[i]]++;

        int len;
        cin >> len;

        messages[i].resize(len);
        set<string> words;
        for (int j = 0; j < len; j++) {
            cin >> messages[i][j];

            words.insert(messages[i][j]);
            dataset_words.insert(messages[i][j]);
        }

        for (const string &word : words) {
            word2classes[word][target[i]]++;
        }
    }

    int m;
    cin >> m;

    vector<vector<string>> messages_test(m);
    vector<set<string>> words(m);
//    vector<long double> target_test(m);

    for (int i = 0; i < m; i++) {
//        cin >> target_test[i];

        int len;
        cin >> len;

        messages_test[i].resize(len);
        for (int j = 0; j < len; j++) {
            cin >> messages_test[i][j];

            words[i].insert(messages_test[i][j]);
        }
    }

//    for (int c : classes) {
//        for (const string &word : dataset_words) {
//            cout << pxc(word, c, word2classes, class2cnt, alpha) << " ";
//        }
//        cout << '\n';
//    }
//    cout << '\n';
//    for (int i : classes) {
//        cout << py(i, class2cnt, n) << " ";
//    }
//    cout << '\n';
//    cout << '\n';

    map<int, map<int, long double>> pcipXci;

    for (int i = 0; i < m; i++) {
        for (int c : classes) {
            long double p = lambda[c - 1] * py(c, class2cnt, n);

            for (auto &word : dataset_words) {
                long double wordProb = pxc(word, c, word2classes, class2cnt, alpha);

                if (words[i].find(word) == words[i].end()) {
                    p *= (1.0 - wordProb);
                } else {
                    p *= wordProb;
                }
            }

            pcipXci[i][c] = p;
        }
    }

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            int c = j + 1;

            if (classes.find(c) == classes.end()) {
                cout << 0 << " ";
                continue;
            }

            long double p = pcipXci[i][c];
            long double div = 0.0;

            for (int c1 : classes) {
                div += pcipXci[i][c1];
            }

            p /= div;

            cout << p << " ";
        }
        cout << '\n';
    }
}