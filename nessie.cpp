#include <iostream>
#include <sstream> 
#include <random>
#include <string>
#include <immintrin.h>
#include <windows.h>

using namespace std;

#define MAX_PLY 64
#define MAX_MOVE 256
#define MATE 32000
#define INF 32001
#define U8 unsigned __int8
#define S16 signed __int16
#define U16 unsigned __int16
#define S32 signed __int32
#define U32 unsigned __int32
#define S64 signed __int64
#define U64 unsigned __int64
#define NAME "Nessie"
#define VERSION "2026-07-17"
#define START_FEN "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
#define RANK_1_BB 0x00000000000000FFULL
#define RANK_2_BB 0x000000000000FF00ULL
#define RANK_3_BB 0x0000000000FF0000ULL
#define RANK_4_BB 0x00000000FF000000ULL
#define RANK_5_BB 0x000000FF00000000ULL
#define RANK_6_BB 0x0000FF0000000000ULL
#define RANK_7_BB 0x00FF000000000000ULL
#define RANK_8_BB 0xFF00000000000000ULL
#define FILE_A_BB 0x0101010101010101ULL
#define FILE_B_BB 0x0202020202020202ULL
#define FILE_C_BB 0x0404040404040404ULL
#define FILE_D_BB 0x0808080808080808ULL
#define FILE_E_BB 0x1010101010101010ULL
#define FILE_F_BB 0x2020202020202020ULL
#define FILE_G_BB 0x4040404040404040ULL
#define FILE_H_BB 0x8080808080808080ULL
#define BIT(square) (1ULL << (square))
#define COUNT(bitboard) _mm_popcnt_u64(bitboard)
#define LSB(bitboard) _tzcnt_u64(bitboard)
#define FLIP_SQ(sq) ((sq)^0x38)
#define FLIP_BB(bitboard) _byteswap_uint64(bitboard)
#define S(mg,eg) ((eg << 16) + mg)

enum Color { WHITE, BLACK, COLOR_NB };
enum PieceType { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, PT_NB };
enum Bound { TT_LOWER, TT_UPPER, TT_EXACT };
enum Phase { MG, EG, PHASE_NB };
enum Term { PASSED = 6, STRUCTURE, TERM_NB };
enum File : int { FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H, FILE_NB };
enum Rank : int { RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8, RANK_NB };
enum Square : int {
	SQ_A1, SQ_B1, SQ_C1, SQ_D1, SQ_E1, SQ_F1, SQ_G1, SQ_H1,
	SQ_A2, SQ_B2, SQ_C2, SQ_D2, SQ_E2, SQ_F2, SQ_G2, SQ_H2,
	SQ_A3, SQ_B3, SQ_C3, SQ_D3, SQ_E3, SQ_F3, SQ_G3, SQ_H3,
	SQ_A4, SQ_B4, SQ_C4, SQ_D4, SQ_E4, SQ_F4, SQ_G4, SQ_H4,
	SQ_A5, SQ_B5, SQ_C5, SQ_D5, SQ_E5, SQ_F5, SQ_G5, SQ_H5,
	SQ_A6, SQ_B6, SQ_C6, SQ_D6, SQ_E6, SQ_F6, SQ_G6, SQ_H6,
	SQ_A7, SQ_B7, SQ_C7, SQ_D7, SQ_E7, SQ_F7, SQ_G7, SQ_H7,
	SQ_A8, SQ_B8, SQ_C8, SQ_D8, SQ_E8, SQ_F8, SQ_G8, SQ_H8,
	SQ_NONE,
	SQ_NB = 64
};
enum Value :int {
	VALUE_ZERO = 0,
	PawnValueMg = 136, PawnValueEg = 208,
	KnightValueMg = 782, KnightValueEg = 865,
	BishopValueMg = 830, BishopValueEg = 918,
	RookValueMg = 1289, RookValueEg = 1378,
	QueenValueMg = 2529, QueenValueEg = 2687
};

static constexpr Color operator++(Color& c) { return c = Color(int(c) + 1); }
static constexpr Square operator++(Square& f) { return f = Square(int(f) + 1); }
static constexpr Value operator+(Value v, int i) { return Value(int(v) + i); }
static constexpr Value operator-(Value v, int i) { return Value(int(v) - i); }
static inline Value& operator+=(Value& v, int i) { return v = v + i; }
static inline Value& operator-=(Value& v, int i) { return v = v - i; }

int PieceValue[PHASE_NB][PT_NB] = {
  { PawnValueMg, KnightValueMg, BishopValueMg, RookValueMg, QueenValueMg },
  { PawnValueEg, KnightValueEg, BishopValueEg, RookValueEg, QueenValueEg }
};

static constexpr File operator++(File& f) { return f = File(int(f) + 1); }
static constexpr File operator~(File& f) { return File(f ^ FILE_H); }
static constexpr Rank operator++(Rank& r) { return r = Rank(int(r) + 1); }

struct Position {
	bool flipped;
	U8 move50;
	U8 castling[4];
	U64 color[COLOR_NB];
	U64 pieces[PT_NB];
	U64 ep;
};

typedef union {
	U32 move;
	struct {
		U8 from;
		U8 to;
		U8 promo;
		U8 flag;
	};
}Move;

struct Stack {
	S16 score;
	Move move;
	Move killer1;
	Move killer2;
};

struct TTEntry {
	U8 flag;
	U8 depth;
	S16 score;
	U64 hash;
	Move move;
};

struct TTEntryEval {
	S16 score;
	U64 hash;
};

struct SearchInfo {
	bool post;
	bool stop;
	int depthLimit;
	S64 timeStart;
	S64 timeLimit;
	U64 nodes;
	U64 nodesLimit;
}info;

struct Options {
	int elo = 2500;
	int eloMin = 0;
	int eloMax = 2500;
	int ttMb = 64;
	string tempo = "16 8";
}options;

Position pos;
int phase = 0;
const int phases[PT_NB] = { 0, 1, 1, 2, 4, 0 };
const int insVal[PT_NB] = { 5, 2, 3, 5, 5, 0 };
int material[PT_NB] = {};
int mx_material[PT_NB + 1] = {};
int outsideFile[PT_NB] = {};
int outsideRank[PT_NB] = {};
int bonus[PT_NB][RANK_NB][FILE_NB] = {};
int tempo = 0;
int contempt = 0;
int LINEAR_REDUCTION[64][64];

int BonusOrg[PT_NB][RANK_NB][int(FILE_NB) / 2] = {
	//int BonusOrg[PT_NB][RANK_NB][4] = {
	  { // Pawn
	   { S(0, 0), S(0,  0), S(0, 0), S(0, 0) },
	   { S(-11,-3), S(7, -1), S(7, 7), S(17, 2) },
	   { S(-16,-2), S(-3,  2), S(23, 6), S(23,-1) },
	   { S(-14, 7), S(-7, -4), S(20,-8), S(24, 2) },
	   { S(-5,13), S(-2, 10), S(-1,-1), S(12,-8) },
	   { S(-11,16), S(-12,  6), S(-2, 1), S(4,16) },
	   { S(-2, 1), S(20,-12), S(-10, 6), S(-2,25) }
	  },
	  { // Knight
	   { S(-169,-105), S(-96,-74), S(-80,-46), S(-79,-18) },
	   { S(-79, -70), S(-39,-56), S(-24,-15), S(-9,  6) },
	   { S(-64, -38), S(-20,-33), S(4, -5), S(19, 27) },
	   { S(-28, -36), S(5,  0), S(41, 13), S(47, 34) },
	   { S(-29, -41), S(13,-20), S(42,  4), S(52, 35) },
	   { S(-11, -51), S(28,-38), S(63,-17), S(55, 19) },
	   { S(-67, -64), S(-21,-45), S(6,-37), S(37, 16) },
	   { S(-200, -98), S(-80,-89), S(-53,-53), S(-32,-16) }
	  },
	  { // Bishop
	   { S(-49,-58), S(-7,-31), S(-10,-37), S(-34,-19) },
	   { S(-24,-34), S(9, -9), S(15,-14), S(1,  4) },
	   { S(-9,-23), S(22,  0), S(-3, -3), S(12, 16) },
	   { S(4,-26), S(9, -3), S(18, -5), S(40, 16) },
	   { S(-8,-26), S(27, -4), S(13, -7), S(30, 14) },
	   { S(-17,-24), S(14, -2), S(-6,  0), S(6, 13) },
	   { S(-19,-34), S(-13,-10), S(7,-12), S(-11,  6) },
	   { S(-47,-55), S(-7,-32), S(-17,-36), S(-29,-17) }
	  },
	  { // Rook
	   { S(-24, 0), S(-15, 3), S(-8, 0), S(0, 3) },
	   { S(-18,-7), S(-5,-5), S(-1,-5), S(1,-1) },
	   { S(-19, 6), S(-10,-7), S(1, 3), S(0, 3) },
	   { S(-21, 0), S(-7, 4), S(-4,-2), S(-4, 1) },
	   { S(-21,-7), S(-12, 5), S(-1,-5), S(4,-7) },
	   { S(-23, 3), S(-10, 2), S(1,-1), S(6, 3) },
	   { S(-11,-1), S(8, 7), S(9,11), S(12,-1) },
	   { S(-25, 6), S(-18, 4), S(-11, 6), S(2, 2) }
	  },
	  { // Queen
	   { S(3,-69), S(-5,-57), S(-5,-47), S(4,-26) },
	   { S(-3,-55), S(5,-31), S(8,-22), S(12, -4) },
	   { S(-3,-39), S(6,-18), S(13, -9), S(7,  3) },
	   { S(4,-23), S(5, -3), S(9, 13), S(8, 24) },
	   { S(0,-29), S(14, -6), S(12,  9), S(5, 21) },
	   { S(-4,-38), S(10,-18), S(6,-12), S(8,  1) },
	   { S(-5,-50), S(6,-27), S(10,-24), S(8, -8) },
	   { S(-2,-75), S(-2,-52), S(1,-43), S(-2,-36) }
	  },
	  { // King
	   { S(272,  0), S(325, 41), S(273, 80), S(190, 93) },
	   { S(277, 57), S(305, 98), S(241,138), S(183,131) },
	   { S(198, 86), S(253,138), S(168,165), S(120,173) },
	   { S(169,103), S(191,152), S(136,168), S(108,169) },
	   { S(145, 98), S(176,166), S(112,197), S(69, 194) },
	   { S(122, 87), S(159,164), S(85, 174), S(36, 189) },
	   { S(87,  40), S(120, 99), S(64, 128), S(25, 141) },
	   { S(64,   5), S(87,  60), S(49,  75), S(0,   75) }
	  }
};

// MobilityBonus[PieceType-2][attacked] contains bonuses for middle and end game,
// indexed by piece type and number of attacked squares in the mobility area.
int MobilityBonus[][32] = {
  { S(-62,-81), S(-53,-56), S(-12,-30), S(-4,-14), S(3,  8), S(13, 15), // Knights
	S(22, 23), S(28, 27), S(33, 33) },
  { S(-48,-59), S(-20,-23), S(16, -3), S(26, 13), S(38, 24), S(51, 42), // Bishops
	S(55, 54), S(63, 57), S(63, 65), S(68, 73), S(81, 78), S(81, 86),
	S(91, 88), S(98, 97) },
  { S(-58,-76), S(-27,-18), S(-15, 28), S(-10, 55), S(-5, 69), S(-2, 82), // Rooks
	S(9,112), S(16,118), S(30,132), S(29,142), S(32,155), S(38,165),
	S(46,166), S(48,169), S(58,171) },
  { S(-39,-36), S(-21,-15), S(3,  8), S(3, 18), S(14, 34), S(22, 54), // Queens
	S(28, 61), S(41, 73), S(43, 79), S(48, 92), S(56, 94), S(60,104),
	S(60,113), S(66,120), S(67,123), S(70,126), S(71,133), S(73,136),
	S(79,140), S(88,143), S(88,148), S(99,166), S(102,170), S(102,175),
	S(106,184), S(109,191), S(113,206), S(116,212) }
};

U64 bbSquare[SQ_NB];
U64 bbKnightAttack[SQ_NB];
U64 bbKingAttack[SQ_NB];
U64 ttMask;
vector<TTEntry> tt;
vector<TTEntryEval> ttEval;
U64 keys[848];
Stack stack[128]{};
S64 hh[2][SQ_NB][SQ_NB]{};
int historyCount = 0;
U64 historyHash[1024]{};
U64 bbDistanceRing[SQ_NB][8];

U64 bishop_table[64][512];
U64 rook_table[64][4096];
U64 bishop_masks[64];
U64 rook_masks[64];

const int rook_deltas[4] = { 8, -8, 1, -1 };
const int bishop_deltas[4] = { 9, -9, 7, -7 };

void UciCommand(string command);

static inline void HHClear() { memset(hh, 0, sizeof(hh)); }
static inline void StackClear() { memset(stack, 0, sizeof(stack)); }
static inline void TTClear() { memset(tt.data(), 0, sizeof(TTEntry) * tt.size()); }
static inline void TTEvalClear() { memset(ttEval.data(), 0, sizeof(TTEntryEval) * ttEval.size()); }
static inline U64 East(const U64 bb) { return (bb << 1) & ~0x0101010101010101ULL; }
static inline U64 West(const U64 bb) { return (bb >> 1) & ~0x8080808080808080ULL; }
static inline U64 North(const U64 bb) { return bb << 8; }
static inline U64 South(const U64 bb) { return bb >> 8; }
static inline U64 NW(const U64 bb) { return (bb << 7) & ~FILE_H_BB; }
static inline U64 NE(const U64 bb) { return (bb << 9) & ~FILE_A_BB; }
static inline U64 SW(const U64 bb) { return (bb >> 9) & ~FILE_H_BB; }
static inline U64 SE(const U64 bb) { return (bb >> 7) & ~FILE_A_BB; }
static inline Rank RankOf(int sq) { return Rank(sq >> 3); }
static inline File FileOf(int sq) { return File(sq & 7); }

static inline U64 GetTimeMs() {
#if defined(_WIN32)
	static LARGE_INTEGER freq;
	static int init = 0;
	if (!init) {
		QueryPerformanceFrequency(&freq);
		init = 1;
	}
	LARGE_INTEGER counter;
	QueryPerformanceCounter(&counter);
	return (U64)((counter.QuadPart * 1000) / freq.QuadPart);
#else
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (U64)(ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
#endif
}

static void InitTT(int mb) {
	U64 ttCount = 1;
	while (ttCount < mb * 1e6 / (sizeof(TTEntry) + sizeof(TTEntryEval)))
		ttCount <<= 1;
	ttMask = ttCount - 1;
	tt.resize(ttCount);
	ttEval.resize(ttCount);
	TTClear();
	TTEvalClear();
}

static int IsValid(int square, int delta) {
	int r = square / 8;
	int c = square % 8;
	int nr = (square + delta) / 8;
	int nc = (square + delta) % 8;
	if (nr < 0 || nr > 7 || nc < 0 || nc > 7) return 0;
	if (delta == 1 || delta == -1) return r == nr;
	if (delta == 9 || delta == -9 || delta == 7 || delta == -7) {
		return (r - nr == 1 || r - nr == -1) && (c - nc == 1 || c - nc == -1);
	}
	return 1;
}

static U64 GenerateSliderAttacks(int square, U64 occupancy, int is_rook) {
	U64 attacks = 0ULL;
	const int* deltas = is_rook ? rook_deltas : bishop_deltas;
	for (int i = 0; i < 4; i++) {
		int target = square;
		while (IsValid(target, deltas[i])) {
			target += deltas[i];
			attacks |= (1ULL << target);
			if (occupancy & (1ULL << target)) break;
		}
	}
	return attacks;
}

static void InitSlidersBmi2() {
	for (int sq = 0; sq < 64; sq++) {
		U64 bishop_mask = 0ULL;
		for (int i = 0; i < 4; i++) {
			int target = sq;
			while (IsValid(target, bishop_deltas[i]) && IsValid(target + bishop_deltas[i], bishop_deltas[i])) {
				target += bishop_deltas[i];
				bishop_mask |= (1ULL << target);
			}
		}
		bishop_masks[sq] = bishop_mask;

		U64 rook_mask = 0ULL;
		for (int i = 0; i < 4; i++) {
			int target = sq;
			while (IsValid(target, rook_deltas[i]) && IsValid(target + rook_deltas[i], rook_deltas[i])) {
				target += rook_deltas[i];
				rook_mask |= (1ULL << target);
			}
		}
		rook_masks[sq] = rook_mask;

		int bishop_bits = COUNT(bishop_mask);
		int bishop_permutations = 1 << bishop_bits;
		for (int i = 0; i < bishop_permutations; i++) {
			U64 occupancy = _pdep_u64(i, bishop_mask);
			U64 attacks = GenerateSliderAttacks(sq, occupancy, 0);
			uint64_t index = _pext_u64(occupancy, bishop_mask);
			bishop_table[sq][index] = attacks;
		}

		int rook_bits = COUNT(rook_mask);
		int rook_permutations = 1 << rook_bits;
		for (int i = 0; i < rook_permutations; i++) {
			U64 occupancy = _pdep_u64(i, rook_mask);
			U64 attacks = GenerateSliderAttacks(sq, occupancy, 1);

			uint64_t index = _pext_u64(occupancy, rook_mask);
			rook_table[sq][index] = attacks;
		}
	}
}

inline U64 BishopAttack(int square, U64 occupancy) {
	U64 mask = bishop_masks[square];
	return bishop_table[square][_pext_u64(occupancy, mask)];
}

inline U64 RookAttack(int square, U64 occupancy) {
	U64 mask = rook_masks[square];
	return rook_table[square][_pext_u64(occupancy, mask)];
}

static bool IsRepetition(Position& pos, U64 hash) {
	int limit = max(0, historyCount - pos.move50);
	for (int n = historyCount - 4; n >= limit; n -= 2)
		if (historyHash[n] == hash)
			return true;
	return false;
}

inline static S64 SqToBb(int sq) {
	if (sq < 0 || sq > 63)
		return 0;
	return 1ULL << sq;
}

static int Distance(int sq1, int sq2) {
	return max(abs(FileOf(sq1) - FileOf(sq2)), abs(RankOf(sq1) - RankOf(sq2)));
};

static int KingDistance(int sq1, int sq2) {
	return min(Distance(sq1, sq2), 5);
}

static int Mg(int score) {
	return (short)score;
}

static int Eg(int score) {
	return (score + 0x8000) >> 16;
}

static int ValueMax(int score) {
	return max(Mg(score), Eg(score));
}

static bool MoreThanOne(U64 b) {
	return b & (b - 1);
}

static void FlipPosition(Position& pos) {
	pos.color[0] = FLIP_BB(pos.color[0]);
	pos.color[1] = FLIP_BB(pos.color[1]);
	for (int i = 0; i < 6; ++i) {
		pos.pieces[i] = FLIP_BB(pos.pieces[i]);
	}
	pos.ep = FLIP_BB(pos.ep);
	swap(pos.color[0], pos.color[1]);
	swap(pos.castling[0], pos.castling[2]);
	swap(pos.castling[1], pos.castling[3]);
	pos.flipped = !pos.flipped;
}

static string SquareToUci(const int sq, const int flip) {
	string str;
	str += 'a' + (sq % 8);
	str += '1' + (flip ? (7 - sq / 8) : (sq / 8));
	return str;
}

static auto MoveToUci(const Move& move, const int flip) {
	string str = SquareToUci(move.from, flip);
	str += SquareToUci(move.to, flip);
	if (move.promo != PT_NB) {
		str += "\0nbrq\0\0"[move.promo];
	}
	return str;
}

static Move UciToMove(string& uci, int flip) {
	Move m;
	m.from = (uci[0] - 'a');
	int f = (uci[1] - '1');
	m.from += 8 * (flip ? 7 - f : f);
	m.to = (uci[2] - 'a');
	f = (uci[3] - '1');
	m.to += 8 * (flip ? 7 - f : f);
	m.promo = PT_NB;
	switch (uci[4]) {
	case 'N':
	case 'n':
		m.promo = KNIGHT;
		break;
	case 'B':
	case 'b':
		m.promo = BISHOP;
		break;
	case 'R':
	case 'r':
		m.promo = ROOK;
		break;
	case 'Q':
	case 'q':
		m.promo = QUEEN;
		break;
	}
	return m;
}

static int PieceTypeOn(const Position& pos, const int sq) {
	const U64 bb = 1ULL << sq;
	for (int i = 0; i < 6; ++i) {
		if (pos.pieces[i] & bb) {
			return i;
		}
	}
	return PT_NB;
}

static void ResetInfo() {
	info.post = true;
	info.stop = false;
	info.nodes = 0;
	info.depthLimit = MAX_PLY;
	info.nodesLimit = 0;
	info.timeLimit = 0;
	info.timeStart = GetTimeMs();
}

template <typename F>
U64 Ray(const U64 bb, const U64 blockers, F f) {
	U64 mask = f(bb);
	mask |= f(mask & ~blockers);
	mask |= f(mask & ~blockers);
	mask |= f(mask & ~blockers);
	mask |= f(mask & ~blockers);
	mask |= f(mask & ~blockers);
	mask |= f(mask & ~blockers);
	mask |= f(mask & ~blockers);
	return mask;
}

static U64 BBKnightAttack(const U64 bb) {
	return (((bb << 15) | (bb >> 17)) & 0x7F7F7F7F7F7F7F7FULL) | (((bb << 17) | (bb >> 15)) & 0xFEFEFEFEFEFEFEFEULL) |
		(((bb << 10) | (bb >> 6)) & 0xFCFCFCFCFCFCFCFCULL) | (((bb << 6) | (bb >> 10)) & 0x3F3F3F3F3F3F3F3FULL);
}

static U64 KnightAttack(const int sq, const U64) {
	return bbKnightAttack[sq];
}

static U64 BBBishopAttack(const U64 bb, const U64 blockers) {
	return Ray(bb, blockers, NW) | Ray(bb, blockers, NE) | Ray(bb, blockers, SW) | Ray(bb, blockers, SE);
}

static U64 BBRookAttack(const U64 bb, const U64 blockers) {
	return Ray(bb, blockers, North) | Ray(bb, blockers, East) | Ray(bb, blockers, South) | Ray(bb, blockers, West);
}

static U64 BBKingAttack(const U64 bb) {
	return (bb << 8) | (bb >> 8) |
		(((bb >> 1) | (bb >> 9) | (bb << 7)) & 0x7F7F7F7F7F7F7F7FULL) |
		(((bb << 1) | (bb << 9) | (bb >> 7)) & 0xFEFEFEFEFEFEFEFEULL);
}

static U64 KingAttack(const int sq, const U64) {
	return bbKingAttack[sq];
}

static bool IsAttacked(const Position& pos, const int sq, const int them = true) {
	const U64 bb = bbSquare[sq];
	const U64 kt = pos.color[them] & pos.pieces[KNIGHT];
	const U64 BQ = pos.pieces[BISHOP] | pos.pieces[QUEEN];
	const U64 RQ = pos.pieces[ROOK] | pos.pieces[QUEEN];
	const U64 pawns = pos.color[them] & pos.pieces[PAWN];
	const U64 pawn_attacks = them ? SW(pawns) | SE(pawns) : NW(pawns) | NE(pawns);
	return (pawn_attacks & bb) | (kt & KnightAttack(sq, 0)) |
		(BishopAttack(sq, pos.color[0] | pos.color[1]) & pos.color[them] & BQ) |
		(RookAttack(sq, pos.color[0] | pos.color[1]) & pos.color[them] & RQ) |
		(KingAttack(sq, 0) & pos.color[them] & pos.pieces[KING]);
}

static auto MakeMove(Position& pos, const Move& move) {
	const int piece = PieceTypeOn(pos, move.from);
	const int captured = PieceTypeOn(pos, move.to);
	const U64 to = bbSquare[move.to];
	const U64 from = bbSquare[move.from];
	pos.move50++;
	if (captured != PT_NB || piece == PAWN)
		pos.move50 = 0;
	pos.color[0] ^= from | to;
	pos.pieces[piece] ^= from | to;
	if (piece == PAWN && to == pos.ep) {
		pos.color[1] ^= to >> 8;
		pos.pieces[PAWN] ^= to >> 8;
	}
	pos.ep = 0x0ULL;
	if (piece == PAWN && move.to - move.from == 16) {
		pos.ep = to >> 8;
	}
	if (captured != PT_NB) {
		pos.color[1] ^= to;
		pos.pieces[captured] ^= to;
	}
	if (piece == KING) {
		const U64 bb = move.to - move.from == 2 ? 0xa0ULL : move.to - move.from == -2 ? 0x9ULL : 0x0ULL;
		pos.color[0] ^= bb;
		pos.pieces[ROOK] ^= bb;
	}
	if (piece == PAWN && move.to >= 56) {
		pos.pieces[PAWN] ^= to;
		pos.pieces[move.promo] ^= to;
	}
	pos.castling[0] &= !((from | to) & 0x90ULL);
	pos.castling[1] &= !((from | to) & 0x11ULL);
	pos.castling[2] &= !((from | to) & 0x9000000000000000ULL);
	pos.castling[3] &= !((from | to) & 0x1100000000000000ULL);
	FlipPosition(pos);
	return !IsAttacked(pos, (int)LSB(pos.color[1] & pos.pieces[KING]), false);
}

static void AddMove(Move* const moves_list, int& num_moves, const U8 from, const U8 to, const U8 promo = PT_NB) {
	Move& move = moves_list[num_moves++];
	move.from = from;
	move.to = to;
	move.promo = promo;
	move.flag = 0;
}

static void generate_pawn_moves(Move* const movelist, int& num_moves, U64 to_mask, const int offset) {
	while (to_mask) {
		const int to = (int)LSB(to_mask);
		to_mask &= to_mask - 1;
		if (to >= 56) {
			AddMove(movelist, num_moves, to + offset, to, QUEEN);
			AddMove(movelist, num_moves, to + offset, to, ROOK);
			AddMove(movelist, num_moves, to + offset, to, BISHOP);
			AddMove(movelist, num_moves, to + offset, to, KNIGHT);
		}
		else
			AddMove(movelist, num_moves, to + offset, to);
	}
}

static void generate_piece_moves(Move* const movelist, int& num_moves, const Position& pos, const int piece, const U64 to_mask, U64(*func)(int, U64)) {
	U64 copy = pos.color[0] & pos.pieces[piece];
	while (copy) {
		const int fr = LSB(copy);
		copy &= copy - 1;
		U64 moves = func(fr, pos.color[0] | pos.color[1]) & to_mask;
		while (moves) {
			const int to = LSB(moves);
			moves &= moves - 1;
			AddMove(movelist, num_moves, fr, to);
		}
	}
}

static int GenerateMoves(const Position& pos, Move* const movelist, const bool only_captures) {
	int num_moves = 0;
	const U64 all = pos.color[0] | pos.color[1];
	const U64 to_mask = only_captures ? pos.color[1] : ~pos.color[0];
	const U64 pawns = pos.color[0] & pos.pieces[PAWN];
	generate_pawn_moves(movelist, num_moves, North(pawns) & ~all & (only_captures ? 0xFF00000000000000ULL : 0xFFFFFFFFFFFF0000ULL), -8);
	if (!only_captures)
		generate_pawn_moves(movelist, num_moves, North(North(pawns & 0xFF00ULL) & ~all) & ~all, -16);
	generate_pawn_moves(movelist, num_moves, NW(pawns) & (pos.color[1] | pos.ep), -7);
	generate_pawn_moves(movelist, num_moves, NE(pawns) & (pos.color[1] | pos.ep), -9);
	generate_piece_moves(movelist, num_moves, pos, KNIGHT, to_mask, KnightAttack);
	generate_piece_moves(movelist, num_moves, pos, BISHOP, to_mask, BishopAttack);
	generate_piece_moves(movelist, num_moves, pos, QUEEN, to_mask, BishopAttack);
	generate_piece_moves(movelist, num_moves, pos, ROOK, to_mask, RookAttack);
	generate_piece_moves(movelist, num_moves, pos, QUEEN, to_mask, RookAttack);
	generate_piece_moves(movelist, num_moves, pos, KING, to_mask, KingAttack);
	if (!only_captures && pos.castling[0] && !(all & 0x60ULL) && !IsAttacked(pos, 4) && !IsAttacked(pos, 5))
		AddMove(movelist, num_moves, 4, 6);
	if (!only_captures && pos.castling[1] && !(all & 0xEULL) && !IsAttacked(pos, 4) && !IsAttacked(pos, 3))
		AddMove(movelist, num_moves, 4, 2);
	return num_moves;
}

static constexpr U64 Attacks(int pt, int sq, U64 blockers) {
	switch (pt) {
	case ROOK:
		return RookAttack(sq, blockers);
	case BISHOP:
		return BishopAttack(sq, blockers);
	case QUEEN:
		return RookAttack(sq, blockers) | BishopAttack(sq, blockers);
	case KNIGHT:
		return KnightAttack(sq, blockers);
	case KING:
		return KingAttack(sq, blockers);
	default:
		return 0;
	}
}

static auto GetHash(const Position& pos) {
	U64 hash = pos.flipped;
	for (S32 p = PAWN; p < PT_NB; ++p) {
		U64 copy = pos.pieces[p] & pos.color[0];
		while (copy) {
			const S32 sq = LSB(copy);
			copy &= copy - 1;
			hash ^= keys[p * 64 + sq];
		}
		copy = pos.pieces[p] & pos.color[1];
		while (copy) {
			const S32 sq = LSB(copy);
			copy &= copy - 1;
			hash ^= keys[(p + 6) * 64 + sq];
		}
	}
	if (pos.ep)
		hash ^= keys[12 * 64 + LSB(pos.ep)];
	hash ^= keys[13 * 64 + pos.castling[0] + pos.castling[1] * 2 + pos.castling[2] * 4 + pos.castling[3] * 8];
	return hash;
}

static bool InputAvailable() {
	static HANDLE hstdin = 0;
	static bool pipe = false;
	unsigned long dw = 0;
	if (!hstdin) {
		hstdin = GetStdHandle(STD_INPUT_HANDLE);
		pipe = !GetConsoleMode(hstdin, &dw);
		if (!pipe)
		{
			SetConsoleMode(hstdin, dw & ~(ENABLE_MOUSE_INPUT | ENABLE_WINDOW_INPUT));
			FlushConsoleInputBuffer(hstdin);
		}
		else
		{
			setvbuf(stdin, NULL, _IONBF, 0);
			setvbuf(stdout, NULL, _IONBF, 0);
		}
	}
	if (pipe)
		PeekNamedPipe(hstdin, 0, 0, 0, &dw, 0);
	else
		GetNumberOfConsoleInputEvents(hstdin, &dw);
	return dw > 1;
}

static bool CheckUp() {
	if ((++info.nodes & 0xffff) == 0) {
		if (info.timeLimit && GetTimeMs() - info.timeStart > info.timeLimit)
			info.stop = true;
		if (info.nodesLimit && info.nodes > info.nodesLimit)
			info.stop = true;
		if (InputAvailable()) {
			string line;
			getline(cin, line);
			UciCommand(line);
		}
	}
	return info.stop;
}

static bool IsPseudolegalMove(const Position& pos, const Move& move) {
	Move moves[256];
	const int num_moves = GenerateMoves(pos, moves, false);
	for (int i = 0; i < num_moves; ++i)
		if (moves[i].move == move.move)
			return true;
	return false;
}

static void PrintPv(const Position& pos, const Move move) {
	if (!IsPseudolegalMove(pos, move))
		return;
	Position npos = pos;
	if (!MakeMove(npos, move))
		return;
	cout << " " << MoveToUci(move, pos.flipped);
	const U64 tt_key = GetHash(npos);
	const TTEntry& tt_entry = tt[tt_key & ttMask];
	if (tt_entry.hash != tt_key || tt_entry.flag == TT_UPPER)
		return;
	if (IsRepetition(npos, tt_key))
		return;
	historyHash[historyCount++] = tt_key;
	PrintPv(npos, tt_entry.move);
	historyCount--;
}

static int Permill() {
	int pm = 0;
	for (int n = 0; n < 1000; n++) {
		if (tt[n].hash)
			pm++;
	}
	return pm;
}

//prints the bitboard
static void PrintBitboard(U64 bb) {
	const char* s = "   +---+---+---+---+---+---+---+---+\n";
	const char* t = "     A   B   C   D   E   F   G   H\n";
	cout << t;
	for (int i = 56; i >= 0; i -= 8) {
		cout << s << " " << i / 8 + 1 << " ";
		for (int x = 0; x < 8; x++) {
			const char* c = 1LL << (i + x) & bb ? "x" : " ";
			cout << "| " << c << " ";
		}
		cout << "| " << i / 8 + 1 << endl;
	}
	cout << s;
	cout << t << endl;
}

//prints the board
static void PrintBoard(Position& pos) {
	Position np = pos;
	if (np.flipped)
		FlipPosition(np);
	const char* s = "   +---+---+---+---+---+---+---+---+\n";
	const char* t = "     A   B   C   D   E   F   G   H\n";
	cout << t;
	for (int i = 56; i >= 0; i -= 8) {
		cout << s << " " << i / 8 + 1 << " ";
		for (int j = 0; j < 8; j++) {
			int sq = i + j;
			int piece = PieceTypeOn(np, sq);
			if (np.color[0] & 1ull << sq)
				cout << "| " << "ANBRQK "[piece] << " ";
			else
				cout << "| " << "anbrqk "[piece] << " ";
		}
		cout << "| " << i / 8 + 1 << endl;
	}
	cout << s;
	cout << t << endl;
	char castling[5] = "KQkq";
	for (int n = 0; n < 4; n++)
		if (!np.castling[n])
			castling[n] = '-';
	printf("side     : %16s\n", pos.flipped ? "black" : "white");
	printf("castling : %16s\n", castling);
	printf("hash     : %16llx\n", GetHash(pos));
}

static int ShrinkNumber(U64 n) {
	if (n < 10000)
		return 0;
	if (n < 10000000)
		return 1;
	if (n < 10000000000)
		return 2;
	return 3;
}

//displays a summary
static void PrintSummary(U64 time, U64 nodes) {
	if (time < 1)
		time = 1;
	U64 nps = (nodes * 1000) / time;
	const char* units[] = { "", "k", "m", "g" };
	int sn = ShrinkNumber(nps);
	U64 p = (U64)pow(10, sn * 3);
	printf("-----------------------------\n");
	printf("Time        : %llu\n", time);
	printf("Nodes       : %llu\n", nodes);
	printf("Nps         : %llu (%llu%s/s)\n", nps, nps / p, units[sn]);
	printf("-----------------------------\n");
}

static int EvalPosition(Position& pos) {
	int score = tempo;
	int insufficient[COLOR_NB]{};
	phase = 0;
	for (int c = 0; c < COLOR_NB; ++c) {
		U64 bbAll = pos.color[0] | pos.color[1];
		const U64 bbPawnsUs = pos.color[0] & pos.pieces[PAWN];
		const U64 bbPawnsEn = pos.color[1] & pos.pieces[PAWN];
		const U64 bbPawnAttack = SE(bbPawnsEn) | SW(bbPawnsEn);
		U64 lowRanks = RANK_2_BB | RANK_3_BB;
		U64 bbBlocked = bbPawnsUs & (South(bbAll) | lowRanks);
		U64 bbMobilityArea = ~(bbBlocked | ((pos.pieces[QUEEN] | pos.pieces[KING]) & pos.color[0]) | bbPawnAttack);
		for (int pt = 0; pt < PT_NB; ++pt) {
			U64 copy = pos.color[0] & pos.pieces[pt];
			while (copy) {
				phase += phases[pt];
				insufficient[c] += insVal[pt];
				const int sq = LSB(copy);
				copy &= copy - 1;
				const int rank = RankOf(sq);
				const int file = FileOf(sq);
				score += bonus[pt][rank][file];
				if ((pt >= KNIGHT) && (pt <= QUEEN)) {
					U64 bbAttacks = Attacks(pt, sq, bbAll);
					score += MobilityBonus[pt - KNIGHT][COUNT(bbAttacks & bbMobilityArea)];
				}
			}

		}
		FlipPosition(pos);
		score = -score;
	}
	phase = min(24, phase);
	int value = (Mg(score) * phase + Eg(score) * (24 - phase)) / 24;
	if (max(insufficient[0], insufficient[1]) < 5)return 0;
	if (insufficient[value < 0] < 4)return 0;
	return value;
}

static int EvalHash(Position& pos, U64 hash) {
	TTEntryEval& ee = ttEval[hash & ttMask];
	if (ee.hash != hash) {
		ee.hash = hash;
		ee.score = EvalPosition(pos);
	}
	return (ee.score * (100 - pos.move50)) / 100;
}

static string StrToLower(string s) {
	transform(s.begin(), s.end(), s.begin(), ::tolower);
	return s;
}

static void SplitStr(const std::string& txt, std::vector<std::string>& vStr, char ch) {
	vStr.clear();
	if (txt == "")
		return;
	size_t pos = txt.find(ch);
	size_t initialPos = 0;
	while (pos != std::string::npos) {
		vStr.push_back(txt.substr(initialPos, pos - initialPos));
		initialPos = pos + 1;
		pos = txt.find(ch, initialPos);
	}
	vStr.push_back(txt.substr(initialPos, min(pos, txt.size()) - initialPos + 1));
}

static void SplitInt(const string& txt, vector<int>& vInt, char ch) {
	vInt.clear();
	vector<string> vs;
	SplitStr(txt, vs, ch);
	for (string s : vs)
		vInt.push_back(stoi(s));
}

static int GetVal(vector<int> v, int i) {
	if (i >= 0 && i < v.size())
		return v[i];
	return 0;
}

static void InitEval() {
	int mg, eg;
	vector<int> split{};
	int elo = options.elo;
	if (elo < options.eloMin)
		elo = options.eloMin;
	if (elo > options.eloMax)
		elo = options.eloMax;
	elo -= options.eloMin;
	int eloRange = options.eloMax - options.eloMin;
	int eloMod = RookValueMg * 2;
	eloMod -= (eloMod * elo) / eloRange;
	for (int pt = PAWN; pt < PT_NB; pt++) {
		mg = PieceValue[0][pt] - eloMod;
		eg = PieceValue[1][pt];
		material[pt] = S(mg, eg);
		mx_material[pt] = max(mg, eg);
	}

	SplitInt(options.tempo, split, ' ');
	mg = GetVal(split, 0);
	eg = GetVal(split, 1);
	tempo = S(mg, eg);

	for (int pt = PAWN; pt < PT_NB; ++pt)
		for (int r = RANK_1; r < RANK_NB; ++r)
			for (int f = FILE_A; f < FILE_NB; ++f){
				int fi = min(int(f), 7 - f);
				bonus[pt][r][f] = material[pt];
				bonus[pt][r][f] += BonusOrg[pt][r][fi];
			}
	for (int s1 = 0; s1 <= 63; ++s1)
		for (int s2 = 0; s2 <= 63; ++s2)
			if (s1 != s2)
			{
				int df = std::abs(FileOf(Square(s1)) - FileOf(Square(s2)));
				int dr = std::abs(RankOf(Square(s1)) - RankOf(Square(s2)));
				int d = max(df, dr);
				bbDistanceRing[s1][d] |= (1ULL << s2);
			}
}

static void PrintInfo(Position& pos, int depth, int score) {
	cout << "info depth " << depth;
	if (abs(score) < MATE - MAX_PLY)
		cout << " score cp " << score;
	else
		cout << " score mate " << (score > 0 ? (MATE - score + 1) >> 1 : -(MATE + score) >> 1);
	const auto elapsed = GetTimeMs() - info.timeStart;
	cout << " time " << elapsed;
	cout << " nodes " << info.nodes;
	cout << " hashfull " << Permill();
	cout << " pv";
	PrintPv(pos, stack[0].move);
	cout << endl;
}

static int ScoreToTT(int score, int ply) {
	if (score >= MATE - MAX_PLY) return score + ply;
	if (score <= -MATE + MAX_PLY) return score - ply;
	return score;
}

static int ScoreFromTT(int score, int ply) {
	if (score >= MATE - MAX_PLY) return score - ply;
	if (score <= -MATE + MAX_PLY) return score + ply;
	return score;
}

static S16 SearchAlpha(Position& pos, int alpha, int beta, int depth, int ply, bool doNull = true) {
	if (CheckUp())
		return 0;
	int  mate_value = MATE - ply;
	if (alpha < -mate_value)
		alpha = -mate_value;
	if (beta > mate_value - 1)
		beta = mate_value - 1;
	if (alpha >= beta)
		return alpha;
	const U64 hash = GetHash(pos);
	const int staticEval = stack[ply].score = EvalHash(pos, hash);
	if (ply >= MAX_PLY - 1)
		return staticEval;
	const bool inCheck = IsAttacked(pos, (int)LSB(pos.color[0] & pos.pieces[KING]));
	if (inCheck)
		depth = max(1, depth + 1);
	bool inQuiescence = depth < 1;
	if (ply && !inQuiescence) {
		if (!(pos.pieces[PAWN] | pos.pieces[ROOK] | pos.pieces[QUEEN]) && COUNT(pos.pieces[BISHOP] | pos.pieces[KNIGHT]) < 2)
			return 0;
		if (pos.move50 > 99 || IsRepetition(pos, hash))
			return 0;
	}
	int ttScore = 0;
	TTEntry& ttEntry = tt[hash & ttMask];
	stack[ply + 1].move.move = 0;
	Move ttMove = stack[ply + 1].move;
	bool inPv = beta - alpha > 1;
	if (ttEntry.hash == hash) {
		ttMove = ttEntry.move;
		ttScore = ScoreFromTT(ttEntry.score, ply);
		if (!inPv && ttEntry.depth >= depth) {
			if (ttEntry.flag == TT_EXACT)return ttScore;
			if (ttEntry.flag == TT_UPPER && ttScore <= alpha)return ttScore;
			if (ttEntry.flag == TT_LOWER && ttScore >= beta)return ttScore;
		}
	}
	else
		depth -= depth > 3;
	if (inQuiescence && alpha < staticEval) {
		alpha = staticEval;
		if (alpha >= beta)
			return beta;
	}
	const int improving = ply > 1 && staticEval > stack[ply - 2].score;
	if (ply && !inQuiescence && !inCheck && !inPv) {

		// Reverse futility pruning
		if (depth < 8) {
			if (staticEval - 71 * (depth - improving) >= beta)
				return staticEval;
			inQuiescence = staticEval + 238 * depth < alpha;
		}

		// Null move pruning
		if (depth > 2 && staticEval >= beta && doNull && pos.color[0] & ~pos.pieces[PAWN] & ~pos.pieces[KING]) {
			Position npos = pos;
			FlipPosition(npos);
			npos.ep = 0x0ULL;
			if (-SearchAlpha(npos, -beta, -alpha, depth - 4 - depth / 5 - min((staticEval - beta) / 196, 3), ply + 1, false) >= beta)
				return beta;
		}
	}
	historyHash[historyCount++] = hash;
	U8 ttFlag = TT_UPPER;
	int legalCount = 0;
	int score;
	int quietCount = 0;
	Move movesList[MAX_MOVE];
	Move quietList[MAX_MOVE];
	S64 scoreList[MAX_MOVE];
	S32 best_score = inQuiescence ? staticEval : -INF;
	const int movesCount = GenerateMoves(pos, movesList, inQuiescence);
	for (int j = 0; j < movesCount; ++j) {
		Move m = movesList[j];
		const int ptSou = PieceTypeOn(pos, m.from);
		const int ptDes = m.promo == PT_NB ? PieceTypeOn(pos, m.to) : m.promo;
		if (m.move == ttMove.move)
			scoreList[j] = BIT(62);
		else if (ptDes != PT_NB)
			scoreList[j] = BIT(54) + ptDes * 10 - ptSou;
		else if (m.move == stack[ply].killer1.move)
			scoreList[j] = BIT(49);
		else if (m.move == stack[ply].killer2.move)
			scoreList[j] = BIT(48);
		else
			scoreList[j] = hh[pos.flipped][m.from][m.to];
	}
	for (int i = 0; i < movesCount; ++i) {
		int bstIdx = i;
		for (int j = i + 1; j < movesCount; ++j)
			if (scoreList[bstIdx] < scoreList[j])
				bstIdx = j;
		Move move = movesList[bstIdx];
		scoreList[bstIdx] = scoreList[i];
		movesList[bstIdx] = movesList[i];

		// Material gain
		const S32 gain = mx_material[move.promo] + mx_material[PieceTypeOn(pos, move.to)];

		// Delta pruning
		if (inQuiescence && !inCheck && staticEval + 50 + mx_material[PieceTypeOn(pos, move.to)] < alpha)
			break;

		// Forward futility pruning
		if (ply > 0 && depth < 8 && !inQuiescence && !inCheck && legalCount && staticEval + 105 * depth + gain < alpha)
			break;

		Position npos = pos;
		if (!MakeMove(npos, move))
			continue;
		if (!legalCount || inCheck || depth < 4)
			score = -SearchAlpha(npos, -beta, -alpha, depth - 1, ply + 1);
		else {
			int r = LINEAR_REDUCTION[min(63, depth)][min(63, legalCount)];
			const int givesCheck = IsAttacked(npos, LSB(npos.color[0] & npos.pieces[KING]), 1);
			if (inPv)
				r--;
			if (inCheck || givesCheck)
				r--;
			if (gain)
				r--;
			if (r > depth - 2)
				r = depth - 2;
			if (r < 0)
				r = 0;
			score = -SearchAlpha(npos, -alpha - 1, -alpha, depth - 1 - r, ply + 1);
			if (r && score > alpha)
				score = -SearchAlpha(npos, -alpha - 1, -alpha, depth - 1, ply + 1);
			if (score > alpha && score < beta)
				score = -SearchAlpha(npos, -beta, -alpha, depth - 1, ply + 1);
		}
		if (info.stop)
			break;
		legalCount++;
		if (best_score < score)best_score = score;
		if (alpha < score) {
			alpha = score;
			ttFlag = TT_EXACT;
			stack[ply].move = move;
			if (!ply && info.post)
				PrintInfo(pos, depth, score);
			if (alpha >= beta) {
				ttFlag = TT_LOWER;
				if (!gain) {
					stack[ply].killer2.move = stack[ply].killer1.move;
					stack[ply].killer1.move = move.move;
				}
				int bonus = depth * depth;
				hh[pos.flipped][move.from][move.to] +=
					bonus - bonus * hh[pos.flipped][move.from][move.to] / 512;
				for (int j = 0; j < quietCount; ++j)
					hh[pos.flipped][quietList[j].from][quietList[j].to] -=
					bonus + bonus * hh[pos.flipped][quietList[j].from][quietList[j].to] / 512;
				break;
			}
		}
		if (!gain)
			quietList[quietCount++] = move;
		if (!inCheck && !inPv && quietCount > (1 + depth * depth) >> (int)!improving)break;
	}
	historyCount--;
	if (info.stop)
		return 0;
	ttEntry.hash = hash;
	if (stack[ply].move.move)
		ttEntry.move.move = stack[ply].move.move;
	ttEntry.depth = max(0, depth);
	ttEntry.score = ScoreToTT(best_score, ply);
	ttEntry.flag = ttFlag;
	if (!legalCount && !inQuiescence)
		return inCheck ? ply - MATE : 0;
	return best_score;
}

static void SearchIteratively(Position& pos) {
	HHClear();
	StackClear();
	TTClear();
	int score = 0;
	int alpha = -MATE;
	int beta = MATE;
	for (int depth = 1; depth <= info.depthLimit; ++depth) {
		int aspH = 16, aspL = 16;
		do {
			if (depth > 4) {
				alpha = score - aspL;
				beta = score + aspH;
			}
			score = SearchAlpha(pos, alpha, beta, depth, 0);
			if (score <= alpha) {
				alpha -= aspL;
				aspL *= 2;
			}
			else if (score >= beta) {
				beta += aspH;
				aspH *= 2;
			}
			else
				break;
		} while (!info.stop);
		if (info.stop)
			break;
		if (info.timeLimit && GetTimeMs() - info.timeStart > info.timeLimit / 2)
			break;
	}
	if (info.post)
		cout << "bestmove " << MoveToUci(stack[0].move, pos.flipped) << endl << flush;
}

static inline void PerftDriver(Position pos, int depth) {
	Move list[256];
	const S32 num_moves = GenerateMoves(pos, list, false);
	for (int n = 0; n < num_moves; n++) {
		Position npos = pos;
		if (!MakeMove(npos, list[n]))
			continue;
		if (depth)
			PerftDriver(npos, depth - 1);
		else
			info.nodes++;
	}
}

static void SetFen(Position& pos, const string& fen) {
	pos.flipped = false;
	pos.ep = 0;
	memset(pos.color, 0, sizeof(pos.color));
	memset(pos.pieces, 0, sizeof(pos.pieces));
	memset(pos.castling, 0, sizeof(pos.castling));
	stringstream stack(fen);
	string word;
	stack >> word;
	int i = 56;
	for (const auto c : word) {
		if (c >= '1' && c <= '8')
			i += c - '1' + 1;
		else if (c == '/')
			i -= 16;
		else {
			const int side = c == 'p' || c == 'n' || c == 'b' || c == 'r' || c == 'q' || c == 'k';
			const int piece = (c == 'p' || c == 'P') ? PAWN
				: (c == 'n' || c == 'N') ? KNIGHT
				: (c == 'b' || c == 'B') ? BISHOP
				: (c == 'r' || c == 'R') ? ROOK
				: (c == 'q' || c == 'Q') ? QUEEN
				: KING;
			pos.color[side] ^= BIT(i);
			pos.pieces[piece] ^= BIT(i);
			i++;
		}
	}
	stack >> word;
	const bool sideBlack = word == "b";
	stack >> word;
	for (const auto c : word) {
		pos.castling[0] |= c == 'K';
		pos.castling[1] |= c == 'Q';
		pos.castling[2] |= c == 'k';
		pos.castling[3] |= c == 'q';
	}
	stack >> word;
	if (word != "-") {
		const int sq = word[0] - 'a' + 8 * (word[1] - '1');
		pos.ep = 1ULL << sq;
	}
	stack >> word;
	pos.move50 = stoi(word);
	if (sideBlack)
		FlipPosition(pos);
}

static void PrintPerformanceHeader() {
	printf("-----------------------------\n");
	printf("ply      time        nodes\n");
	printf("-----------------------------\n");
}

//start benchmark
static void UciBench(Position& pos) {
	ResetInfo();
	PrintPerformanceHeader();
	SetFen(pos, START_FEN);
	info.depthLimit = 0;
	info.post = false;
	S64 elapsed = 0;
	while (elapsed < 8000) {
		++info.depthLimit;
		SearchIteratively(pos);
		elapsed = GetTimeMs() - info.timeStart;
		printf("%2d. %8llu %12llu\n", info.depthLimit, elapsed, info.nodes);
	}
	PrintSummary(GetTimeMs() - info.timeStart, info.nodes);
}

//start performance test
static void UciPerformance(Position& pos) {
	ResetInfo();
	PrintPerformanceHeader();
	int depth = 0;
	SetFen(pos, START_FEN);
	while (GetTimeMs() - info.timeStart < 4000) {
		info.nodes = 0;
		info.timeStart = GetTimeMs();
		PerftDriver(pos, depth++);
		printf("%2d. %8llu %12llu\n", depth, GetTimeMs() - info.timeStart, info.nodes);
	}
	PrintSummary(GetTimeMs() - info.timeStart, info.nodes);
}

static int ScoreToValue(int score) {
	int mgWeight = phase;
	int egWeight = 24 - mgWeight;
	return (mgWeight * Mg(score) + egWeight * Eg(score)) / 24;
}

static void ParsePosition(Position& pos, string command) {
	string fen = START_FEN;
	stringstream stack(command);
	string token;
	stack >> token;
	if (token != "position")
		return;
	stack >> token;
	if (token == "startpos")
		stack >> token;
	else if (token == "fen") {
		fen = "";
		while (stack >> token && token != "moves")
			fen += token + " ";
		fen.pop_back();
	}
	SetFen(pos, fen);
	historyCount = 0;
	while (stack >> token) {
		Move m = UciToMove(token, pos.flipped);
		if (PieceTypeOn(pos, m.to) != PT_NB || PieceTypeOn(pos, m.from) == PAWN)
			historyCount = 0;
		historyHash[historyCount++] = GetHash(pos);
		MakeMove(pos, m);
	}
}

static void ParseGo(Position& pos, string command) {
	stringstream ss(command);
	string token;
	ss >> token;
	if (token != "go")
		return;
	ResetInfo();
	int wtime = 0;
	int btime = 0;
	int winc = 0;
	int binc = 0;
	int movestogo = 32;
	while (ss >> token) {
		if (token == "wtime")
			ss >> wtime;
		else if (token == "btime")
			ss >> btime;
		else if (token == "winc")
			ss >> winc;
		else if (token == "binc")
			ss >> binc;
		else if (token == "movestogo")
			ss >> movestogo;
		else if (token == "movetime")
			ss >> info.timeLimit;
		else if (token == "depth")
			ss >> info.depthLimit;
		else if (token == "nodes")
			ss >> info.nodesLimit;
	}
	int time = pos.flipped ? btime : wtime;
	int inc = pos.flipped ? binc : winc;
	if (time)
		info.timeLimit = max(1, min(time / movestogo + inc, time / 2));
	SearchIteratively(pos);
}

void UciCommand(string command) {
	if (command.empty())
		return;
	stringstream stack(command);
	string token;
	stack >> token;
	if (token == "uci")
	{
		cout << "id name " << NAME << endl;
		cout << "option name UCI_Elo type spin default " << options.eloMax << " min " << options.eloMin << " max " << options.eloMax << endl;
		cout << "option name hash type spin default " << options.ttMb << " min 1 max 1000" << endl;
		cout << "uciok" << endl;
	}
	else if (token == "isready")cout << "readyok" << endl;
	else if (token == "ucinewgame")TTEvalClear();
	else if (token == "position")
		ParsePosition(pos, command);
	else if (token == "go")
		ParseGo(pos, command);
	else if (token == "setoption") {
		stack >> token;
		stack >> token;
		token = StrToLower(token);
		if (token == "uci_elo") {
			stack >> token;
			stack >> options.elo;
			InitEval();
		}
		else if (token == "hash") {
			stack >> token;
			stack >> options.ttMb;
			InitTT(options.ttMb);
		}
	}
	else if (token == "bench")UciBench(pos);
	else if (token == "perft")UciPerformance(pos);
	else if (token == "print")PrintBoard(pos);
	else if (token == "stop")info.stop = true;
	else if (token == "quit")exit(0);
}

static void Init() {
	for (int depth = 1; depth < 64; depth++)
		for (int legal = 1; legal < 64; legal++)
			LINEAR_REDUCTION[depth][legal] = (int)(0.75 + log(depth) * log(legal) / 2.25);
}

static void InitBitboards() {
	for (int sq = 0; sq < 64; ++sq) {
		U64 bb = BIT(sq);
		bbSquare[sq] = bb;
		bbKnightAttack[sq] = BBKnightAttack(bb);
		bbKingAttack[sq] = BBKingAttack(bb);
	}
}

static void InitHash() {
	mt19937_64 r;
	for (U64& k : keys)
		k = r();
}

int main(const int argc, const char** argv) {
	Init();
	InitBitboards();
	InitHash();
	InitEval();
	InitTT(options.ttMb);
	InitSlidersBmi2();
	setbuf(stdout, NULL);
	cout << NAME << " " << VERSION << endl;
	SetFen(pos, START_FEN);
	string line;
	while (true) {
		getline(cin, line);
		UciCommand(line);
	}
}
