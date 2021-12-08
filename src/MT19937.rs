// Copyright (c) 2015,2017 rust-mersenne-twister developers
// Copyright (c) 2021 Tony Yang, Arabesque AI
//
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. All files in the project carrying such notice may not be copied,
// modified, or distributed except according to those terms.

// Original source: https://github.com/dcrewi/rust-mersenne-twister
// Modified to be inline with the scikit-learn implementation

use std::num::Wrapping;

const N: usize = 624;
const M: usize = 397;
const ONE: Wrapping<u32> = Wrapping(1);
const MATRIX_A: Wrapping<u32> = Wrapping(0x9908b0df);
const UPPER_MASK: Wrapping<u32> = Wrapping(0x80000000);
const LOWER_MASK: Wrapping<u32> = Wrapping(0x7fffffff);

/// The 32-bit flavor of the Mersenne Twister pseudorandom number
/// generator.
pub struct MT19937 {
    idx: usize,
    state: [Wrapping<u32>; N],
}

pub const UNINITIALIZED: MT19937 = MT19937 {
    idx: 0,
    state: [Wrapping(0); N],
};

impl MT19937 {
    /// Create a new Mersenne Twister random number generator using a provided seed
    #[inline]
    pub fn from_seed(seed: u32) -> MT19937 {
        let mut ret = UNINITIALIZED;
        ret.reseed(seed);
        ret
    }

    #[inline]
    fn fill_next_state(&mut self) {
        for i in 0..N - M {
            let x =
                (self.state[i] & UPPER_MASK) | (self.state[i + 1] & LOWER_MASK);
            self.state[i] =
                self.state[i + M] ^ (x >> 1) ^ ((x & ONE) * MATRIX_A);
        }
        for i in N - M..N - 1 {
            let x =
                (self.state[i] & UPPER_MASK) | (self.state[i + 1] & LOWER_MASK);
            self.state[i] =
                self.state[i + M - N] ^ (x >> 1) ^ ((x & ONE) * MATRIX_A);
        }
        let x = (self.state[N - 1] & UPPER_MASK) | (self.state[0] & LOWER_MASK);
        self.state[N - 1] =
            self.state[M - 1] ^ (x >> 1) ^ ((x & ONE) * MATRIX_A);
        self.idx = 0;
    }

    pub fn reseed(&mut self, seed: u32) {
        self.idx = N;
        self.state[0] = Wrapping(seed);
        for i in 1..N {
            self.state[i] = Wrapping(1812433253)
                * (self.state[i - 1] ^ (self.state[i - 1] >> 30))
                + Wrapping(i as u32);
        }
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        // Failing this check indicates that, somehow, the structure
        // was not initialized.
        // debug_assert!(self.idx != 0);
        if self.idx >= N {
            self.fill_next_state();
        }
        let Wrapping(x) = self.state[self.idx];
        self.idx += 1;
        temper(x)
    }

    #[inline]
    pub fn bounded_rand_int(&mut self, max: u32) -> u32 {
        // "LibSVM / LibLinear Original way" - make a 31bit positive
        // random number and use modulo to make it fit in the range
        // self.next_u32() % max

        // "Better way": tweaked Lemire post-processor
        // from http://www.pcg-random.org/posts/bounded-rands.html
        let mut x = self.next_u32();
        let mut m = x as u64 * max as u64;
        let mut l = m as u32;
        if l < max {
            let mut t = max.wrapping_neg();
            if t >= max {
                t -= max;
                if t >= max {
                    t %= max
                }
            }
            while l < t {
                x = self.next_u32();
                m = x as u64 * max as u64;
                l = m as u32;
            }
        }
        (m >> 32) as u32
    }
}

#[inline]
fn temper(mut x: u32) -> u32 {
    x ^= x >> 11;
    x ^= (x << 7) & 0x9d2c5680;
    x ^= (x << 15) & 0xefc60000;
    x ^= x >> 18;
    x
}
