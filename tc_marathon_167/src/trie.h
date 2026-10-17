// Prefix trie over the upper-case dictionary.
#ifndef MM167_TRIE_H
#define MM167_TRIE_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace mm167 {
struct Trie {
    struct Node {
        std::array<int, 26> next{};
        uint32_t mask = 0;
        bool terminal = false;
    };
    std::vector<Node> nodes{1};
    void add(const std::string& word) {
        int p = 0;
        for (char ch : word) {
            int c = (ch >= 'a' && ch <= 'z') ? ch-'a' : ch-'A';
            if (!nodes[p].next[c]) {
                int id = static_cast<int>(nodes.size());
                nodes[p].next[c] = id;
                nodes[p].mask |= 1u << c;
                nodes.emplace_back();
            }
            p = nodes[p].next[c];
        }
        nodes[p].terminal = true;
    }
    bool contains(const std::string& word) const {
        int p = 0;
        for (char ch : word) {
            p = nodes[p].next[ch-'A'];
            if (!p) return false;
        }
        return nodes[p].terminal;
    }
};
}
#endif
