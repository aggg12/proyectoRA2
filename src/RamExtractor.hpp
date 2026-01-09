#ifndef RAM_EXTRACTOR_HPP
#define RAM_EXTRACTOR_HPP

#include <ale_interface.hpp>
#include <vector>
#include <iostream>
#include <iomanip>

struct GameState {
    int cannonX;
    int temperature; 
    
    int enemy1X;
    int enemy2X;
    int enemy3X;

    // Projectile Data (Unified)
    int projectileActive; // 1 if either vertical or horizontal projectile is active
    int projectileX;      // From 0x91
    int projectileY;      // From 0x6E (vertical) or assumed constant/different for fireball

    int levelType; 
    int lives;
    int score;
};

class RamExtractor {
public:
    static GameState extract(ALEInterface& ale);
    static void printGameState(const GameState& state);

private:
    static int getRam(ALEInterface& ale, int offset);
    static int getLevelType(int levelNumber);
};

#endif
