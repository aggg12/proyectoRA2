#ifndef EVONET_NEAT_INNOVATION_DATABASE_HPP
#define EVONET_NEAT_INNOVATION_DATABASE_HPP

#include <map>
#include <utility>

namespace evonet {
namespace neat {

struct InnovationKey {
    int inNode;
    int outNode;

    bool operator<(const InnovationKey& other) const {
        if (inNode != other.inNode) return inNode < other.inNode;
        return outNode < other.outNode;
    }
};

class InnovationDatabase {
private:
    std::map<InnovationKey, int> history;
    int currentInnovation;
    int currentNodeId;

public:
    InnovationDatabase() : currentInnovation(0), currentNodeId(0) {}

    int getInnovation(int inNode, int outNode) {
        InnovationKey key{inNode, outNode};
        auto it = history.find(key);
        if (it != history.end()) {
            return it->second;
        }
        int newInnov = ++currentInnovation;
        history[key] = newInnov;
        return newInnov;
    }

    int createNodeId() {
        return ++currentNodeId;
    }

    void setCurrentNodeId(int nextId) {
        currentNodeId = nextId;
    }

    int getCurrentNodeId() const {
        return currentNodeId;
    }
};

} // namespace neat
} // namespace evonet

#endif
