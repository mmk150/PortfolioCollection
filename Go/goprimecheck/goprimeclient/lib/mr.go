package lib

import (
	"math/big"
	"math/rand"
	"time"
)

// MillerRabin performs the Miller-Rabin primality test
// n is the number to test
// k is the number of iterations (higher k means more accuracy)
func MillerRabin(n *big.Int, k int) bool {
	if n.Cmp(big.NewInt(2)) < 0 {
		return false
	}
	if n.Cmp(big.NewInt(2)) == 0 {
		return true
	}
	if new(big.Int).Mod(n, big.NewInt(2)).Cmp(big.NewInt(0)) == 0 {
		return false
	}

	// Write n-1 as 2^r * d where d is odd
	d := new(big.Int).Sub(n, big.NewInt(1))
	r := 0
	for new(big.Int).Mod(d, big.NewInt(2)).Cmp(big.NewInt(0)) == 0 {
		d.Div(d, big.NewInt(2))
		r++
	}

	rnd := rand.New(rand.NewSource(time.Now().UnixNano()))

	for i := 0; i < k; i++ {

		a := new(big.Int)
		for {
			a.Rand(rnd, new(big.Int).Sub(n, big.NewInt(4)))
			a.Add(a, big.NewInt(2))
			if a.Cmp(n) < 0 {
				break
			}
		}

		// x = a^d mod n
		x := new(big.Int).Exp(a, d, n)

		if x.Cmp(big.NewInt(1)) == 0 || x.Cmp(new(big.Int).Sub(n, big.NewInt(1))) == 0 {
			continue
		}

		composite := true
		for j := 0; j < r-1; j++ {
			// x = x^2 mod n
			x.Exp(x, big.NewInt(2), n)
			if x.Cmp(new(big.Int).Sub(n, big.NewInt(1))) == 0 {
				composite = false
				break
			}
			if x.Cmp(big.NewInt(1)) == 0 {
				return false
			}
		}
		if composite {
			return false
		}
	}

	return true
}
