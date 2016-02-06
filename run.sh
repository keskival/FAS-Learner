#!/bin/bash
rm report.json
th learner.lua --name 0.9 --learningRate 0.9 --minLR 0.9 &> lr09.out
th learner.lua --name 0.8 --learningRate 0.8 --minLR 0.8 &> lr08.out
th learner.lua --name 0.7 --learningRate 0.7 --minLR 0.7 &> lr07.out
th learner.lua --name 0.6 --learningRate 0.6 --minLR 0.6 &> lr06.out
th learner.lua --name 0.5 --learningRate 0.5 --minLR 0.5 &> lr05.out
th learner.lua --name 0.4 --learningRate 0.4 --minLR 0.4 &> lr04.out
th learner.lua --name 0.3 --learningRate 0.3 --minLR 0.3 &> lr03.out
th learner.lua --name 0.2 --learningRate 0.2 --minLR 0.2 &> lr02.out
th learner.lua --name 0.1 --learningRate 0.1 --minLR 0.1 &> lr01.out
