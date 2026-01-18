# Compilador y Flags
CXX = g++
CXXFLAGS = -std=c++17 -O3 -Iinclude -Wall -Wextra

# Directorios
DIR_BASIC = examples/01_basics
DIR_DL    = examples/02_deep_learning
DIR_NEAT  = examples/03_neat

# --- OBJETIVOS (TARGETS) ---

# Básicos
T_XOR       = main_xor
T_SINE      = main_sine
T_NEAT_XOR  = main_neat_xor
T_NEAT_SINE = main_neat_sine

# Deep Learning (Redes Fijas)
T_DL_TRAIN = dl_train
T_DL_LOAD  = dl_load
T_DL_TEST  = dl_test

# NEAT (Redes Evolutivas)
T_NEAT_TRAIN  = neat_train
T_NEAT_LOAD   = neat_load
T_NEAT_TEST   = neat_test

ALL_TARGETS = $(T_XOR) $(T_SINE) $(T_NEAT_XOR) $(T_NEAT_SINE) \
              $(T_DL_TRAIN) $(T_DL_LOAD) $(T_DL_TEST) \
              $(T_NEAT_TRAIN) $(T_NEAT_LOAD) $(T_NEAT_TEST)

# --- REGLAS PRINCIPALES ---

all: $(ALL_TARGETS)
	@echo "=========================================="
	@echo "  Todo compilado correctamente."
	@echo "  Usa 'make help' para ver comandos."
	@echo "=========================================="

help:
	@echo "COMANDOS DISPONIBLES:"
	@echo "  make all           -> Compila todo"
	@echo "  make clean         -> Borra ejecutables y .o"
	@echo "  make distclean     -> Borra ejecutables y modelos (.txt)"
	@echo ""
	@echo "--- BASICS (Carpeta 01_basics) ---"
	@echo "  make run_xor          -> Standard NN XOR"
	@echo "  make run_sine         -> Standard NN Sine"
	@echo "  make run_neat_xor     -> NEAT XOR"
	@echo "  make run_neat_sine    -> NEAT Sine"
	@echo ""
	@echo "--- DEEP LEARNING (Carpeta 02_deep_learning) ---"
	@echo "  make train_dl      -> Entrenar (main_digits.cpp)"
	@echo "  make load_dl       -> Cargar (main_load_digits.cpp)"
	@echo "  make test_dl       -> Testear (main_test_robustness.cpp)"
	@echo ""
	@echo "--- NEAT (Carpeta 03_neat) ---"
	@echo "  make train_neat    -> Entrenar (main_digits.cpp)"
	@echo "  make load_neat     -> Cargar (main_load_digits.cpp)"
	@echo "  make test_neat     -> Testear (main_test_robustness.cpp)"

# --- COMPILACIÓN ---

# 1. Básicos
$(T_XOR): $(DIR_BASIC)/main_xor.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

$(T_SINE): $(DIR_BASIC)/main_sine.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

$(T_NEAT_XOR): $(DIR_BASIC)/main_neat_xor.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

$(T_NEAT_SINE): $(DIR_BASIC)/main_neat_sine.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# 2. Deep Learning
$(T_DL_TRAIN): $(DIR_DL)/main_digits.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

$(T_DL_LOAD): $(DIR_DL)/main_load_digits.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

$(T_DL_TEST): $(DIR_DL)/main_test_robustness.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# 3. NEAT
$(T_NEAT_TRAIN): $(DIR_NEAT)/main_digits.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

$(T_NEAT_LOAD): $(DIR_NEAT)/main_load_digits.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

$(T_NEAT_TEST): $(DIR_NEAT)/main_test_robustness.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# --- EJECUCIÓN ---

# Básicos
run_xor: $(T_XOR)
	./$(T_XOR)

run_sine: $(T_SINE)
	./$(T_SINE)

run_neat_xor: $(T_NEAT_XOR)
	./$(T_NEAT_XOR)

run_neat_sine: $(T_NEAT_SINE)
	./$(T_NEAT_SINE)

# Deep Learning Shortcuts
train_dl: $(T_DL_TRAIN)
	./$(T_DL_TRAIN)

load_dl: $(T_DL_LOAD)
	./$(T_DL_LOAD)

test_dl: $(T_DL_TEST)
	./$(T_DL_TEST)

# NEAT Shortcuts
train_neat: $(T_NEAT_TRAIN)
	./$(T_NEAT_TRAIN)

load_neat: $(T_NEAT_LOAD)
	./$(T_NEAT_LOAD)

test_neat: $(T_NEAT_TEST)
	./$(T_NEAT_TEST)

# --- LIMPIEZA ---

clean:
	rm -f $(ALL_TARGETS)
	@echo "Ejecutables eliminados."

distclean: clean
	rm -f *_model.txt
	@echo "Modelos (.txt) eliminados."