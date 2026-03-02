// Threadgroup-optimized Poseidon permutation
// Caches both MDS constants and round constants in threadgroup memory to reduce device memory bandwidth
//
// Memory usage per threadgroup:
// - MDS constants: 9 * 8 bytes = 72 bytes (MDS_FREQ blocks)
// - Round constants: 30 * 12 * 8 bytes = 2880 bytes
// - Total: ~3KB, well within Metal's 32KB threadgroup memory limit

#ifndef poseidon_goldilocks_threadgroup
#define poseidon_goldilocks_threadgroup
#include <metal_stdlib>
#include "goldilocks.metal"
#include "u128.h.metal"
#include "poseidon_goldilocks_mds.metal"
#include "poseidon_fast_partial_constants.metal"

using namespace metal;
namespace GoldilocksField {

// Number of round constants: 30 rounds * 12 state elements
constant uint POSEIDON_NUM_ROUNDS = 30;
constant uint POSEIDON_STATE_SIZE = 12;
constant uint POSEIDON_RC_TOTAL = POSEIDON_NUM_ROUNDS * POSEIDON_STATE_SIZE; // 360

// MDS frequency-domain constants (9 values total)
// Block 1: 3 values, Block 2: 3+3 values, Block 3: 3 values
constant uint MDS_CONST_TOTAL = 12; // 3 + 6 + 3 = 12 (for alignment, we use 12)

// MDS constant raw values for threadgroup caching
// MDS_FREQ_BLOCK_ONE = {16, 32, 16}
// MDS_FREQ_BLOCK_TWO.a = {2, -4, 16}, MDS_FREQ_BLOCK_TWO.b = {-1, 1, 1}
// MDS_FREQ_BLOCK_THREE = {-1, -8, 2}
constant long MDS_CONST_RAW[12] = {
    // Block 1 (indices 0-2)
    16, 32, 16,
    // Block 2a (indices 3-5)
    2, -4, 16,
    // Block 2b (indices 6-8)
    -1, 1, 1,
    // Block 3 (indices 9-11)
    -1, -8, 2
};

// Round constants stored as raw ulongs for threadgroup caching
// These must match POSEIDON_RC12 from poseidon_goldilocks.metal
constant ulong POSEIDON_RC_RAW[360] = {
    // Round 0
    13080132714287612933UL, 8594738767457295063UL, 12896916465481390516UL, 1109962092811921367UL,
    16216730422861946898UL, 10137062673499593713UL, 15292064466732465823UL, 17255573294985989181UL,
    14827154241873003558UL, 2846171647972703231UL, 16246264663680317601UL, 14214208087951879286UL,
    // Round 1
    9667108687426275457UL, 6470857420712283733UL, 14103331940138337652UL, 11854816473550292865UL,
    3498097497301325516UL, 7947235692523864220UL, 11110078701231901946UL, 16384314112672821048UL,
    15404405912655775739UL, 14077880830714445579UL, 9555554662709218279UL, 13859595358210603949UL,
    // Round 2
    16859897325061800066UL, 17685474420222222349UL, 17858764734618734949UL, 9410011022665866671UL,
    12495243629579414666UL, 12416945298171515742UL, 5776666812364270983UL, 6314421662864060481UL,
    7402742471423223171UL, 982536713192432718UL, 17321168865775127905UL, 2934354895005980211UL,
    // Round 3
    10567510598607410195UL, 8135543733717919110UL, 116353493081713692UL, 8029688163494945618UL,
    9003846637224807585UL, 7052445132467233849UL, 9645665432288852853UL, 5446430061030868787UL,
    16770910634346036823UL, 17708360571433944729UL, 4661556288322237631UL, 11977051899316327985UL,
    // Round 4
    4378616569090929672UL, 3334807502817538491UL, 8019184735943344966UL, 2395043908812246395UL,
    6558421058331732611UL, 11735894060727326369UL, 8143540538889204488UL, 5991753489563751169UL,
    12235918791502088007UL, 2880312033702687139UL, 18224748115308382355UL, 18070411013125314165UL,
    // Round 5
    8156487614120951180UL, 10615269510047010719UL, 12489426404754222075UL, 5055279340069995710UL,
    7231927319780248664UL, 2602078848106763799UL, 12445944369334781425UL, 3978905923892496205UL,
    16711272944329818038UL, 10439032361227108922UL, 15110119871725214866UL, 821141790655890946UL,
    // Round 6
    11073536380651186235UL, 4866839313097607757UL, 13118391689513956636UL, 14527674973762312380UL,
    7612751959265567999UL, 6808090907814178161UL, 6899703779492644997UL, 3664666286336986826UL,
    783179505424462608UL, 8990689241814097697UL, 9646603555412825679UL, 7351246026167205041UL,
    // Round 7
    16970959813722173256UL, 15735726858241466429UL, 10347018221892268419UL, 12195545878449322889UL,
    7423314197114049891UL, 14908016116973904153UL, 5840340122527363265UL, 17740311462440614128UL,
    815306421953744623UL, 17456357368219253949UL, 6982651076559329072UL, 11970987324614963868UL,
    // Round 8
    8167785008538063246UL, 9483259819397403968UL, 954550221664291548UL, 10339565171024313256UL,
    8651171084286500102UL, 16974445528003515956UL, 15104530047940621190UL, 103271880867179718UL,
    14654666245504492663UL, 12445769555936887967UL, 11250582358051997490UL, 6730977207490590241UL,
    // Round 9
    15919951556166196935UL, 4423540216573360915UL, 16317664700341473511UL, 4723997214951767765UL,
    10098756619006575500UL, 3223149401237667964UL, 6870494874300767682UL, 2902095711130291898UL,
    7159372652788439733UL, 11500508372997952671UL, 13348148181479462670UL, 12729401155983882093UL,
    // Round 10
    15021242795466053388UL, 3802990509227527157UL, 4665459515680145682UL, 13165553315407675603UL,
    6496364397926233172UL, 12800832566287577810UL, 9737592377590267426UL, 8687131091302514939UL,
    1488200421755445892UL, 11004377668730991641UL, 13516338734600228410UL, 2953581820660217936UL,
    // Round 11
    3505040783153922951UL, 3710332827435113697UL, 15414874040873320221UL, 8602547649919482301UL,
    13971349938398812007UL, 187239246702636066UL, 12886019973971254144UL, 4512274763990493707UL,
    2986635507805503192UL, 2315252455709119454UL, 12537995864054210246UL, 2039491936479859267UL,
    // Round 12
    1558644089185031256UL, 4074089203264759305UL, 2522268501749395707UL, 3414760436185256196UL,
    17420887529146466921UL, 2817020417938125001UL, 16538346563888261485UL, 5592270336833998770UL,
    16876602064684906232UL, 1793025614521516343UL, 2178510518148748532UL, 2726440714374752509UL,
    // Round 13
    6502946837278398021UL, 15816362857667988792UL, 12997958454165692924UL, 5314892854495903792UL,
    15533907063555687782UL, 12312015675698548715UL, 14140016464013350248UL, 16325589062962838690UL,
    6796145646370327654UL, 1168753512742361735UL, 4100789820704709368UL, 15947554381540469177UL,
    // Round 14
    8597377839806076919UL, 9704018824195918000UL, 12763288618765762688UL, 17249257732622847695UL,
    1998710993415069759UL, 923759906393011543UL, 1271051229666811593UL, 17822362132088738077UL,
    11797234543722669271UL, 5864538787265942447UL, 15975583211110506970UL, 7258516085733671960UL,
    // Round 15
    17999926471875633100UL, 635992114476018166UL, 17205047318256576347UL, 17384900867876315312UL,
    16484825562915784226UL, 16694130609036138894UL, 10575069350371260875UL, 8330575162062887277UL,
    6212375704691932880UL, 15965138197626618226UL, 14285453069600046939UL, 10005163510208402517UL,
    // Round 16
    885298637936952595UL, 541790758138118921UL, 5985203084790372993UL, 4685030219775483721UL,
    1411106851304815020UL, 11290732479954096478UL, 208280581124868513UL, 10979018648467968495UL,
    8600643745023338215UL, 3477453626867126061UL, 6428436309340258604UL, 5695415667275657934UL,
    // Round 17
    15952065508715623490UL, 15571300830419767248UL, 17259785660502616862UL, 4298425495274316083UL,
    9023601070579319352UL, 7353589709321807492UL, 2988848909076209475UL, 10439527789422046135UL,
    6097734044161429459UL, 1113429873817861476UL, 1639063372386966591UL, 7863102812716788759UL,
    // Round 18
    216040220732135364UL, 14252611488623712688UL, 9543395466794536974UL, 2714461051639810934UL,
    2588317208781407279UL, 15458529123534594916UL, 15748417817551040856UL, 16414455697114422951UL,
    13378164466674639511UL, 13894319928411294675UL, 5032680892090751540UL, 17201338494743078916UL,
    // Round 19
    4397422800601932505UL, 11285062031581972327UL, 7309354640676468207UL, 10457152817239331848UL,
    8855911538863247046UL, 4301853449821814398UL, 13001502396339103326UL, 10218424535115580246UL,
    8628244713920681895UL, 17410423622514037261UL, 14080683768439215375UL, 11453161143447188100UL,
    // Round 20
    16761509772042181939UL, 6688821660695954082UL, 12083434295263160416UL, 8540021431714616589UL,
    6891616215679974226UL, 10229217098454812721UL, 3292165387203778711UL, 6090113424998243490UL,
    13431780521962358660UL, 6061081364215809883UL, 16792066504222214142UL, 16134314044798124799UL,
    // Round 21
    17070233710126619765UL, 6915716851370550800UL, 9505009849073026581UL, 6422700465081897153UL,
    17977653991560529185UL, 5800870252836247255UL, 12096124733159345520UL, 7679273623392321940UL,
    17835783910585744964UL, 2478664878205754377UL, 1720314468413114967UL, 10376757819003248056UL,
    // Round 22
    10376377187857634245UL, 13344930747504284997UL, 11579281865160153596UL, 10300256980048736962UL,
    378765236515040565UL, 11412420941557253424UL, 12931662470734252786UL, 43018908376346374UL,
    3589810689190160071UL, 4688229274750659741UL, 13688957436484306091UL, 11424740943016984272UL,
    // Round 23
    16001900718237913960UL, 5548469743008097574UL, 14584404916672178680UL, 3396622135873576824UL,
    7861729246871155992UL, 16112271126908045545UL, 16988163966860016012UL, 273641680619529493UL,
    15222677154027327363UL, 4070328078309830604UL, 13520458500363296391UL, 8235111705801363015UL,
    // Round 24
    5575990058472514138UL, 2751301609188252989UL, 6478598528223547074UL, 386565553848556638UL,
    9417729078939938713UL, 15204315939835727483UL, 14942015033780606261UL, 18369423901636582012UL,
    4715338437538604447UL, 6840590980607806319UL, 5535471161490539014UL, 5341328005359029952UL,
    // Round 25
    1475161295215894444UL, 7999197814297036636UL, 2984233088665867938UL, 3097746028144832229UL,
    8849530863480031517UL, 7464920943249009773UL, 3802996844641460514UL, 6284458522545927646UL,
    2307388003445002779UL, 4461479354745457623UL, 1649739722664588460UL, 3008391274160432867UL,
    // Round 26
    5142217010456550622UL, 1775580461722730120UL, 161694268822794344UL, 1518963253808031703UL,
    16475258091652710137UL, 119575899007375159UL, 1275863735937973999UL, 16539412514520642374UL,
    2303365191438051950UL, 6435126839960916075UL, 17794599201026020053UL, 13847097589277840330UL,
    // Round 27
    16645869274577729720UL, 8039205965509554440UL, 4788586935019371140UL, 15129007200040077746UL,
    2055561615223771341UL, 4149731103701412892UL, 10268130195734144189UL, 13406631635880074708UL,
    11429218277824986203UL, 15773968030812198565UL, 16050275277550506872UL, 11858586752031736643UL,
    // Round 28
    8927746344866569756UL, 11802068403177695792UL, 157833420806751556UL, 4698875910749767878UL,
    1616722774788291698UL, 3990951895163748090UL, 16758609224720795472UL, 3045571693290741477UL,
    9281634245289836419UL, 13517688176723875370UL, 7961395585333219380UL, 1606574359105691080UL,
    // Round 29
    17564372683613562171UL, 4664015225343144418UL, 6133721340680280128UL, 2667022304383014929UL,
    12316557761857340230UL, 10375614850625292317UL, 8141542666379135068UL, 9185476451083834432UL,
    4991072365274649547UL, 17398204971778820365UL, 16127888338958422584UL, 13586792051317758204UL
};

// Load round constants into threadgroup memory
// Call with first threads in the threadgroup (lid < 360)
inline void load_round_constants_tg(
    uint lid,
    uint num_threads,
    threadgroup ulong* tg_rc
) {
    // Cooperatively load all 360 round constants
    for (uint i = lid; i < POSEIDON_RC_TOTAL; i += num_threads) {
        tg_rc[i] = POSEIDON_RC_RAW[i];
    }
}

// Load MDS constants into threadgroup memory
// Call with first threads in the threadgroup (lid < 12)
inline void load_mds_constants_tg(
    uint lid,
    uint num_threads,
    threadgroup long* tg_mds
) {
    // Cooperatively load all 12 MDS constants
    for (uint i = lid; i < MDS_CONST_TOTAL; i += num_threads) {
        tg_mds[i] = MDS_CONST_RAW[i];
    }
}

// Load both round constants and MDS constants
// tg_memory layout: [0..359] = round constants, [360..371] = MDS constants
inline void load_all_constants_tg(
    uint lid,
    uint num_threads,
    threadgroup ulong* tg_rc,
    threadgroup long* tg_mds
) {
    load_round_constants_tg(lid, num_threads, tg_rc);
    load_mds_constants_tg(lid, num_threads, tg_mds);
}

// Add round constants from threadgroup memory
inline void poseidon_add_rc_tg(thread Fp* p2_state, int roundIndex, threadgroup ulong* tg_rc) {
    uint base = roundIndex * POSEIDON_STATE_SIZE;
    #pragma unroll
    for (int i = 0; i < 12; i++) {
        p2_state[i] = p2_state[i] + Fp(tg_rc[base + i]);
    }
}

inline void poseidon_sbox_all_tg(thread Fp* p2_state) {
    #pragma unroll
    for (int i = 0; i < 12; i++) {
        p2_state[i] = p2_state[i].pow7();
    }
}

// Pair struct for complex operations
template <class t1, class t2>
struct tg_pair {
    t1 a;
    t2 b;
};

// FFT helper functions (same as in poseidon_goldilocks_mds.metal)
inline ulong2 ifft2_real_tg(long2 in) {
    return ulong2((ulong)(in.x + in.y), (ulong)(in.x - in.y));
}

inline ulong4 ifft4_real_tg(long4 in) {
    ulong2 z0 = ifft2_real_tg(long2(in.x + in.w, in.y));
    ulong2 z1 = ifft2_real_tg(long2(in.x - in.w, -in.z));
    return ulong4(z0.x, z1.x, z0.y, z1.y);
}

inline long2 fft2_real_tg(ulong2 in) {
    return long2((long)(in.x + in.y), (long)in.x - (long)in.y);
}

inline long4 fft4_real_tg(ulong4 in) {
    long2 z0 = fft2_real_tg(ulong2(in.x, in.z));
    long2 z1 = fft2_real_tg(ulong2(in.y, in.w));
    return long4(z0.x + z1.x, z0.y, -z1.y, z0.x - z1.x);
}

// MDS block operations using threadgroup-cached constants
inline long3 block1_tg(long3 in, threadgroup long* tg_mds) {
    // tg_mds[0..2] = MDS_FREQ_BLOCK_ONE = {16, 32, 16}
    long3 b1 = long3(tg_mds[0], tg_mds[1], tg_mds[2]);
    return long3(
        in.x * b1.x + in.y * b1.z + in.z * b1.y,
        in.x * b1.y + in.y * b1.x + in.z * b1.z,
        in.x * b1.z + in.y * b1.y + in.z * b1.x
    );
}

inline tg_pair<long3, long3> block2_tg(tg_pair<long3, long3> in, threadgroup long* tg_mds) {
    // tg_mds[3..5] = MDS_FREQ_BLOCK_TWO.a = {2, -4, 16}
    // tg_mds[6..8] = MDS_FREQ_BLOCK_TWO.b = {-1, 1, 1}
    long3 b2a = long3(tg_mds[3], tg_mds[4], tg_mds[5]);
    long3 b2b = long3(tg_mds[6], tg_mds[7], tg_mds[8]);

    long x0s = in.a.x + in.b.x;
    long x1s = in.a.y + in.b.y;
    long x2s = in.a.z + in.b.z;
    long y0s = b2a.x + b2b.x;
    long y1s = b2a.y + b2b.y;
    long y2s = b2a.z + b2b.z;

    long2 m0 = long2(in.a.x * b2a.x, in.b.x * b2b.x);
    long2 m1 = long2(in.a.y * b2a.z, in.b.y * b2b.z);
    long2 m2 = long2(in.a.z * b2a.y, in.b.z * b2b.y);
    long z0r = (m0.x - m0.y) + (x1s * y2s - m1.x - m1.y) + (x2s * y1s - m2.x - m2.y);
    long z0i = (x0s * y0s - m0.x - m0.y) + (-m1.x + m1.y) + (-m2.x + m2.y);

    m0 = long2(in.a.x * b2a.y, in.b.x * b2b.y);
    m1 = long2(in.a.y * b2a.x, in.b.y * b2b.x);
    m2 = long2(in.a.z * b2a.z, in.b.z * b2b.z);
    long z1r = (m0.x - m0.y) + (m1.x - m1.y) + (x2s * y2s - m2.x - m2.y);
    long z1i = (x0s * y1s - m0.x - m0.y) + (x1s * y0s - m1.x - m1.y) + (-m2.x + m2.y);

    m0 = long2(in.a.x * b2a.z, in.b.x * b2b.z);
    m1 = long2(in.a.y * b2a.y, in.b.y * b2b.y);
    m2 = long2(in.a.z * b2a.x, in.b.z * b2b.x);
    long z2r = (m0.x - m0.y) + (m1.x - m1.y) + (m2.x - m2.y);
    long z2i = (x0s * y2s - m0.x - m0.y) + (x1s * y1s - m1.x - m1.y) + (x2s * y0s - m2.x - m2.y);

    return { .a = long3(z0r, z1r, z2r), .b = long3(z0i, z1i, z2i) };
}

inline long3 block3_tg(long3 in, threadgroup long* tg_mds) {
    // tg_mds[9..11] = MDS_FREQ_BLOCK_THREE = {-1, -8, 2}
    long3 b3 = long3(tg_mds[9], tg_mds[10], tg_mds[11]);
    return long3(
        in.x * b3.x - in.y * b3.z - in.z * b3.y,
        in.x * b3.y + in.y * b3.x - in.z * b3.z,
        in.x * b3.z + in.y * b3.y + in.z * b3.x
    );
}

// MDS multiply using threadgroup-cached constants
inline void mds_multiply_freq_tg(unsigned long state[12], threadgroup long* tg_mds) {
    long4 u0 = fft4_real_tg(ulong4(state[0], state[3], state[6], state[9]));
    long4 u1 = fft4_real_tg(ulong4(state[1], state[4], state[7], state[10]));
    long4 u2 = fft4_real_tg(ulong4(state[2], state[5], state[8], state[11]));

    long3 v0 = block1_tg(long3(u0.x, u1.x, u2.x), tg_mds);
    tg_pair<long3, long3> v1 = block2_tg({ .a = long3(u0.y, u1.y, u2.y), .b = long3(u0.z, u1.z, u2.z) }, tg_mds);
    long3 v2 = block3_tg(long3(u0.w, u1.w, u2.w), tg_mds);

    ulong4 s0 = ifft4_real_tg(long4(v0.x, v1.a.x, v1.b.x, v2.x));
    ulong4 s1 = ifft4_real_tg(long4(v0.y, v1.a.y, v1.b.y, v2.y));
    ulong4 s2 = ifft4_real_tg(long4(v0.z, v1.a.z, v1.b.z, v2.z));

    state[0] = s0.x; state[1] = s1.x; state[2] = s2.x;
    state[3] = s0.y; state[4] = s1.y; state[5] = s2.y;
    state[6] = s0.z; state[7] = s1.z; state[8] = s2.z;
    state[9] = s0.w; state[10] = s1.w; state[11] = s2.w;
}

// Apply MDS layer using threadgroup-cached constants
inline void apply_mds_freq_tg(thread Fp* shared, unsigned local_state_offset, threadgroup long* tg_mds) {
    unsigned long state_l[12];
    unsigned long state_h[12];

    #pragma unroll
    for (unsigned j = 0; j < 12; j++) {
        Fp element = shared[local_state_offset + j];
        unsigned long s = (unsigned long)element;
        state_l[j] = s & 0xFFFFFFFF;
        state_h[j] = s >> 32;
    }

    mds_multiply_freq_tg(state_l, tg_mds);
    mds_multiply_freq_tg(state_h, tg_mds);

    u128 s = u128(state_l[0]) + (u128(state_h[0]) << 32);
    s.accumulate_mul_2_ulong(static_cast<ulong>(shared[0]), 8);
    ulong reduced = reduce128(s.high, s.low);
    shared[local_state_offset] = Fp(reduced < GOLDILOCKS_PRIME ? reduced : (reduced - GOLDILOCKS_PRIME));

    #pragma unroll
    for (unsigned j = 1; j < 12; j++) {
        s = u128(state_l[j]) + (u128(state_h[j]) << 32);
        reduced = reduce128(s.high, s.low);
        shared[local_state_offset + j] = Fp(reduced < GOLDILOCKS_PRIME ? reduced : (reduced - GOLDILOCKS_PRIME));
    }
}

// MDS layer using threadgroup-cached constants
inline void poseidon_mds_layer_tg(thread Fp* p2_state, threadgroup long* tg_mds) {
    apply_mds_freq_tg(p2_state, 0, tg_mds);
}

// Full round using all threadgroup-cached constants
inline void poseidon_full_round_tg_full(thread Fp* p2_state, int roundIndex, threadgroup ulong* tg_rc, threadgroup long* tg_mds) {
    poseidon_add_rc_tg(p2_state, roundIndex, tg_rc);
    poseidon_sbox_all_tg(p2_state);
    poseidon_mds_layer_tg(p2_state, tg_mds);
}

// Partial round using all threadgroup-cached constants
inline void poseidon_partial_round_tg_full(thread Fp* p2_state, int roundIndex, threadgroup ulong* tg_rc, threadgroup long* tg_mds) {
    poseidon_add_rc_tg(p2_state, roundIndex, tg_rc);
    p2_state[0] = p2_state[0].pow7();
    poseidon_mds_layer_tg(p2_state, tg_mds);
}

// Poseidon permutation using ALL threadgroup-cached constants (RC + MDS)
// Requires: both tg_rc and tg_mds have been loaded and barrier synced
inline void poseidon_permute_tg_full(thread Fp* p2_state, threadgroup ulong* tg_rc, threadgroup long* tg_mds) {
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        poseidon_full_round_tg_full(p2_state, i, tg_rc, tg_mds);
    }

    #pragma unroll
    for (int i = 4; i < 26; i++) {
        poseidon_partial_round_tg_full(p2_state, i, tg_rc, tg_mds);
    }

    #pragma unroll
    for (int i = 26; i < 30; i++) {
        poseidon_full_round_tg_full(p2_state, i, tg_rc, tg_mds);
    }
}

// ============================================================
// Fast partial round functions (ported from poseidon.rs)
// ============================================================

// Threadgroup memory layout for fast partial constants:
// Offset 0: FAST_PARTIAL_FIRST_RC[12]
// Offset 12: FAST_PARTIAL_RC[22]
// Offset 34: FAST_PARTIAL_INIT_MATRIX[121]
// Offset 155: FAST_PARTIAL_W_HATS[242]
// Offset 397: FAST_PARTIAL_VS[242]
// Total: 639 ulongs

// Load fast partial constants into threadgroup memory cooperatively
inline void load_fast_partial_constants_tg(
    uint lid,
    uint num_threads,
    threadgroup ulong* tg_fp
) {
    for (uint i = lid; i < FAST_PARTIAL_CONST_TOTAL; i += num_threads) {
        if (i < 12) {
            tg_fp[i] = FAST_PARTIAL_FIRST_RC[i];
        } else if (i < 34) {
            tg_fp[i] = FAST_PARTIAL_RC[i - 12];
        } else if (i < 155) {
            tg_fp[i] = FAST_PARTIAL_INIT_MATRIX[i - 34];
        } else if (i < 397) {
            tg_fp[i] = FAST_PARTIAL_W_HATS[i - 155];
        } else {
            tg_fp[i] = FAST_PARTIAL_VS[i - 397];
        }
    }
}

// Add first-round constants to all state elements
inline void partial_first_constant_layer_tg(thread Fp* p2_state, threadgroup ulong* tg_fp) {
    // tg_fp[0..11] = FAST_PARTIAL_FIRST_RC
    #pragma unroll
    for (int i = 0; i < 12; i++) {
        p2_state[i] = p2_state[i] + Fp(tg_fp[i]);
    }
}

// Apply 11x11 initial matrix to state[1..11], state[0] passes through
inline void mds_partial_layer_init_tg(thread Fp* p2_state, threadgroup ulong* tg_fp) {
    // tg_fp[34..154] = FAST_PARTIAL_INIT_MATRIX (11x11, row-major)
    // Initial matrix has first row/column = [1, 0, ..., 0]
    // result[0] = state[0]
    // result[c] = sum(INIT_MATRIX[r-1][c-1] * state[r] for r=1..11) for c=1..11

    Fp result[12];
    result[0] = p2_state[0];

    #pragma unroll
    for (int c = 1; c < 12; c++) {
        Fp acc = Fp(0);
        #pragma unroll
        for (int r = 1; r < 12; r++) {
            uint idx = 34 + (uint)(r - 1) * 11 + (uint)(c - 1);
            acc = acc + Fp(tg_fp[idx]) * p2_state[r];
        }
        result[c] = acc;
    }

    #pragma unroll
    for (int i = 0; i < 12; i++) {
        p2_state[i] = result[i];
    }
}

// Fast partial MDS for round r:
//   result[0] = MDS_M00 * state[0] + sum(W_HATS[r][j] * state[j+1] for j=0..10)
//   result[i] = VS[r][i-1] * state[0] + state[i]  for i=1..11
inline void mds_partial_layer_fast_tg(thread Fp* p2_state, int round, threadgroup ulong* tg_fp) {
    // W_HATS start at offset 155, each round has 11 values
    uint w_hat_base = 155 + (uint)round * 11;
    // VS start at offset 397, each round has 11 values
    uint vs_base = 397 + (uint)round * 11;

    // Compute result[0] = MDS_M00 * state[0] + w_hat . state[1..11]
    Fp s0 = p2_state[0];
    Fp d = Fp(MDS_M00) * s0;

    #pragma unroll
    for (int j = 0; j < 11; j++) {
        d = d + Fp(tg_fp[w_hat_base + (uint)j]) * p2_state[j + 1];
    }

    // Compute result[i] = VS[r][i-1] * state[0] + state[i] for i=1..11
    #pragma unroll
    for (int i = 1; i < 12; i++) {
        p2_state[i] = Fp(tg_fp[vs_base + (uint)(i - 1)]) * s0 + p2_state[i];
    }

    p2_state[0] = d;
}

// Optimized Poseidon permutation with fast partial rounds
// Requires: tg_rc, tg_mds, tg_fp all loaded and barrier-synced
inline void poseidon_permute_tg_fast_partial(
    thread Fp* p2_state,
    threadgroup ulong* tg_rc,
    threadgroup long* tg_mds,
    threadgroup ulong* tg_fp
) {
    // 4 initial full rounds (use original round constants 0-3)
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        poseidon_full_round_tg_full(p2_state, i, tg_rc, tg_mds);
    }

    // Fast partial rounds
    partial_first_constant_layer_tg(p2_state, tg_fp);
    mds_partial_layer_init_tg(p2_state, tg_fp);

    #pragma unroll
    for (int i = 0; i < 22; i++) {
        // S-box on state[0] only
        p2_state[0] = p2_state[0].pow7();
        // Add fast partial round constant to state[0]
        // tg_fp[12..33] = FAST_PARTIAL_RC
        p2_state[0] = p2_state[0] + Fp(tg_fp[12 + i]);
        // Fast partial MDS
        mds_partial_layer_fast_tg(p2_state, i, tg_fp);
    }

    // 4 final full rounds (use original round constants 26-29)
    #pragma unroll
    for (int i = 26; i < 30; i++) {
        poseidon_full_round_tg_full(p2_state, i, tg_rc, tg_mds);
    }
}

// Legacy functions that only use RC caching (for backward compatibility)
inline void poseidon_full_round_tg(thread Fp* p2_state, int roundIndex, threadgroup ulong* tg_rc) {
    poseidon_add_rc_tg(p2_state, roundIndex, tg_rc);
    poseidon_sbox_all_tg(p2_state);
    // Use device-memory MDS (from poseidon_goldilocks_mds.metal included elsewhere)
    apply_mds_freq(p2_state, 0);
}

inline void poseidon_partial_round_tg(thread Fp* p2_state, int roundIndex, threadgroup ulong* tg_rc) {
    poseidon_add_rc_tg(p2_state, roundIndex, tg_rc);
    p2_state[0] = p2_state[0].pow7();
    apply_mds_freq(p2_state, 0);
}

// Legacy permutation using only RC caching
inline void poseidon_permute_tg(thread Fp* p2_state, threadgroup ulong* tg_rc) {
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        poseidon_full_round_tg(p2_state, i, tg_rc);
    }

    #pragma unroll
    for (int i = 4; i < 26; i++) {
        poseidon_partial_round_tg(p2_state, i, tg_rc);
    }

    #pragma unroll
    for (int i = 26; i < 30; i++) {
        poseidon_full_round_tg(p2_state, i, tg_rc);
    }
}

}
#endif
