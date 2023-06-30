function experiment(N) {
    let change = 0 //, noChange = 0
    for (let i = 0; i < N; i++) {
        const award = makeAward()
        const randChoose = randomChoose3()
        change += changeDoor(award, randChoose)
        // noChange += notChangeDoor(award, randChoose)
    }
    console.log(change / N)
}

function makeAward() {
    const award = [0, 0, 0]
    const randAward = randomChoose3()
    award[randAward] = 1
    return award
}

function hostOpen(award, firstChoose) {
    const awardIndex = award.findIndex(n => n == 1)
    let hostOpenIndex = 0
    if (awardIndex == firstChoose) {
        while (true) {
            const k = randomChoose3()
            if (k != awardIndex) {
                hostOpenIndex = k
                break
            }
        }
    } else {
        while (true) {
            const k = randomChoose3()
            if (k != awardIndex && k != firstChoose) {
                hostOpenIndex = k
                break
            }
        }
    }
    return hostOpenIndex
}

function changeDoor(award, firstChoose) {
    // const hostOpenIndex = hostOpen(award, firstChoose)
    return award[firstChoose] == 0
}

function notChangeDoor(award, firstChoose) {
    // const hostOpenIndex = hostOpen(award, firstChoose)
    return award[firstChoose] == 1
}

function randomChoose3() {
    return Math.floor(Math.random() * 3)
}

experiment(100)
experiment(1000)
experiment(10000)