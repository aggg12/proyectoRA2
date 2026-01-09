#include "RamExtractor.hpp"

int RamExtractor::getRam(ALEInterface& ale, int offset) {
    return ale.getRAM().get(offset);
}

int RamExtractor::getLevelType(int level) {
    if (level == 0 || level == 4 || level == 8 || level == 12 || level == 13) return 0;
    if (level == 1 || level == 5 || level == 9) return 1;
    if (level == 2 || level == 6 || level == 10) return 2;
    if (level == 3 || level == 7 || level == 11 || level >= 14) return 3;
    return 0;
}

GameState RamExtractor::extract(ALEInterface& ale) {
    GameState state;

    int rawCannon = getRam(ale, 0x10);
    state.cannonX = (rawCannon & 0x0F) * 2 + ((rawCannon & 0x80) >> 7);

    int t1 = getRam(ale, 0x9C);
    int t2 = getRam(ale, 0x9D);
    int temp = 0;
    if (t1 > 0xC0) {
        int val1 = t1 & 0x3F;
        while (val1) { temp += (val1 & 1); val1 >>= 1; }
    }
    if (t2 > 0) {
        int val2 = t2;
        while (val2) { temp += (val2 & 1); val2 >>= 1; }
    }
    state.temperature = temp;

    int enemyXAddr[] = {0xA1, 0xA2, 0xA3};
    int enemyAliveAddr[] = {0xB6, 0xB7, 0xB8};
    int* targetVars[] = {&state.enemy1X, &state.enemy2X, &state.enemy3X};

    for(int i=0; i<3; ++i) {
        int status = getRam(ale, enemyAliveAddr[i]);
        if (status == 192) {
            int rawX = getRam(ale, enemyXAddr[i]);
            *targetVars[i] = (rawX & 0x0F) * 2 + ((rawX & 0x80) >> 7);
        } else {
            *targetVars[i] = -1;
        }
    }

    int vActive = getRam(ale, 0x4B) & 0x80;
    int hActive = getRam(ale, 0x96) & 0x80;
    
    state.projectileActive = (vActive || hActive) ? 1 : 0;
    
    int rawProjX = getRam(ale, 0x91);
    state.projectileX = (rawProjX & 0x0F) * 2 + ((rawProjX & 0x80) >> 7);
    
    if (hActive) {
        state.projectileY = 180; 
    } else {
        state.projectileY = getRam(ale, 0x6E);
    }

    int levelNum = getRam(ale, 0xE8);
    state.levelType = getLevelType(levelNum);

    state.lives = getRam(ale, 0x65);
    int s1 = getRam(ale, 0x80);
    int s2 = getRam(ale, 0x81);
    int s3 = getRam(ale, 0x82);
    auto bcd = [](int val) { return ((val >> 4) & 0x0F) * 10 + (val & 0x0F); };
    state.score = bcd(s1) * 10000 + bcd(s2) * 100 + bcd(s3);

    return state;
}

void RamExtractor::printGameState(const GameState& state) {
    std::cout << "\033[2J\033[1;1H"; 
    std::cout << "=== ASSAULT RAM EXTRACTOR (Unified) ===" << std::endl;
    std::cout << "Score: " << state.score << " | Lives: " << state.lives << " | Level Type: " << state.levelType << std::endl;
    std::cout << "---------------------------------------" << std::endl;
    std::cout << "Cannon X: " << state.cannonX << " | Temp: " << state.temperature << "/14" << std::endl;
    std::cout << "---------------------------------------" << std::endl;
    std::cout << "Enemies (X):" << std::endl;
    std::cout << "  E1: " << std::setw(2) << state.enemy1X 
              << " | E2: " << std::setw(2) << state.enemy2X 
              << " | E3: " << std::setw(2) << state.enemy3X << std::endl;
    std::cout << "---------------------------------------" << std::endl;
    std::cout << "Projectile:" << std::endl;
    std::cout << "  Active: " << state.projectileActive << std::endl;
    std::cout << "  X Pos : " << state.projectileX << std::endl;
    std::cout << "  Y Pos : " << state.projectileY << " (255=off)" << std::endl;
    std::cout << "=======================================" << std::endl << std::flush;
}
