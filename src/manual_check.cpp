#include <iostream>
#include <ale_interface.hpp>
#include "RamExtractor.hpp"

#ifdef __USE_SDL
  #include <SDL.h>
  #include <poll.h>
#endif

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Uso: ./manual_check <rom_file>" << std::endl;
        return 1;
    }

    ALEInterface ale;
    ale.setInt("random_seed", 123);
    ale.setBool("display_screen", true); 
    ale.setBool("sound", true);
    ale.loadROM(argv[1]);

    std::cout << "=== MODO COMPROBACION MANUAL ===" << std::endl;
    std::cout << "Controles en Terminal:" << std::endl;
    std::cout << "  [ENTER] - Avanzar 1 Frame (Accion NOOP)" << std::endl;
    std::cout << "  'a'     - Izquierda" << std::endl;
    std::cout << "  'd'     - Derecha" << std::endl;
    std::cout << "  'w'     - Disparar (Fuego)" << std::endl;
    std::cout << "  's'     - Abajo" << std::endl;
    std::cout << "  'e'     - Disparar + Derecha" << std::endl;
    std::cout << "  'z'     - Disparar + Izquierda" << std::endl;
    std::cout << "  'q'     - Salir" << std::endl;
    std::cout << "===============================" << std::endl;

    ActionVect legal_actions = ale.getLegalActionSet();
    std::cout << "Acciones legales detectadas:" << std::endl;
    for(size_t i=0; i < legal_actions.size(); ++i) {
        std::cout << "  ID " << legal_actions[i] << " : " << action_to_string(legal_actions[i]) << std::endl;
    }

#ifndef __USE_SDL
    while (!ale.game_over()) {
        GameState state = RamExtractor::extract(ale);
        RamExtractor::printGameState(state);

        std::cout << "\nComando (a=left, d=right, w=fire, e=r+fire, z=l+fire, ENTER=step, q=quit) > ";
        char cmd = std::cin.get();
        if(cmd != '\n') { while(std::cin.get() != '\n'); }

        Action action = PLAYER_A_NOOP;
        int frames_to_run = 1;

        if (cmd == 'q') break;
        else if (cmd == 'a') { action = PLAYER_A_LEFT; frames_to_run = 5; }
        else if (cmd == 'd') { action = PLAYER_A_RIGHT; frames_to_run = 5; }
        else if (cmd == 'e') { action = PLAYER_A_RIGHTFIRE; frames_to_run = 5; }
        else if (cmd == 'z') { action = PLAYER_A_LEFTFIRE; frames_to_run = 5; }
        else if (cmd == 'w') { 
            action = PLAYER_A_UP;
            frames_to_run = 10; 
        }

        std::cout << ">> Accion enviada: " << action << " (" << (int)action << ") por " << frames_to_run << " frames." << std::endl;

        for(int i=0; i<frames_to_run; ++i) {
            ale.act(action); 
        }
    }
#else
    while (!ale.game_over()) {
        GameState state = RamExtractor::extract(ale);
        RamExtractor::printGameState(state);

        std::cout << "\nComando (a=left, d=right, w=fire, e=r+fire, z=l+fire, ENTER=step, q=quit) > " << std::flush;
        
        char cmd = 0;
        bool input_processed = false;

        while (!input_processed) {
            struct pollfd fds;
            fds.fd = 0; 
            fds.events = POLLIN;
            int ret = poll(&fds, 1, 20);

            if (ret > 0) {
                cmd = std::cin.get();
                if(cmd != '\n') { while(std::cin.get() != '\n'); }
                input_processed = true;
            } else {
                SDL_Event event;
                while (SDL_PollEvent(&event)) {
                    if (event.type == SDL_QUIT) {
                        cmd = 'q';
                        input_processed = true;
                    }
                }
            }
        }

        Action action = PLAYER_A_NOOP;
        int frames_to_run = 1; 

        if (cmd == 'q') break;
        else if (cmd == 'a') { action = PLAYER_A_LEFT; frames_to_run = 5; }
        else if (cmd == 'd') { action = PLAYER_A_RIGHT; frames_to_run = 5; }
        else if (cmd == 'e') { action = PLAYER_A_RIGHTFIRE; frames_to_run = 5; }
        else if (cmd == 'z') { action = PLAYER_A_LEFTFIRE; frames_to_run = 5; }
        else if (cmd == 'w') { 
            action = PLAYER_A_UP;
            frames_to_run = 10; 
        }

        std::cout << ">> Accion enviada: " << action << " (" << (int)action << ") por " << frames_to_run << " frames." << std::endl;

        for(int i=0; i<frames_to_run; ++i) {
            ale.act(action);
        }
    }
#endif
    return 0;
}