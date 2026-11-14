# LAJU in simple words

## What it is

LAJU is an AI that designs a bus network for Kelantan. You tell it how many buses there are. It decides where the routes go, where the stops are, and how often each bus comes.

## The goal

Make Kelantan a public-transport state, where people choose the bus instead of the car or motorbike.

The AI is judged on how many people choose to ride, not only on how many live near a stop. Being near a stop is a minimum it must meet (at least 25% of people), not the goal.

## How it works

1. **The map.** LAJU reads Kelantan's real roads and where people live. It picks key stops and joins them with the quickest roads. It also adds stops on roads that carry many people but had none, like Jalan Pintu Geng.
2. **The people.** It guesses where people want to travel (home, school, work, mall, clinic) and whether they would take the bus or their own vehicle. Today almost everyone prefers the motorbike. The guess is tuned to BAS.MY's real ridership of about 2,800 boardings a day.
3. **The AI.** A neural network draws the routes one stop at a time. It practises on 300 different views of Kelantan, and learns by comparing its own tries and copying the better ones. A second method, a genetic algorithm (GA), polishes the final design.
4. **The score.** Each design gets a score from the number of riders, how straight and short the routes are, and whether it stays within the bus budget (64 buses, like BAS.MY today). Lower is better.
5. **The exam.** The design is checked on real trips: can people reach school, a clinic, a mall or the train within a set time? It is compared with BAS.MY's real 16 routes.

## The website

A Next.js site (port 3001) that talks to a Python engine (port 8000). It shows the routes on a map, the results of each design, and the AI practising live.

## The GPS logger

A separate program that records where BAS.MY's buses really are, every 30 seconds. Later this gives the AI real road speeds instead of guesses. It does not log when BAS.MY's feed stops updating.

## Where it stands (10 Nov 2026)

- With the same 64 buses, the AI's designs carry more riders than BAS.MY's real routes and reach more places.
- The AI stops improving after about 1.5 hours of practice each night. Fixes planned: keep a notebook of its best designs, shrink the learning step over the night, and let the GA improve the notebook.
- Today the weak point is coverage: about 23% of people live near a stop, against BAS.MY's 30%. The 25% minimum in the new score is meant to fix that.

## Next steps

1. Put real GPS speeds into the road map.
2. Give each bus a price, so the AI can choose how many routes and buses to run, without adding buses forever.
3. Show how many riders 64, 100, 200 and 400 buses would win.
4. Ask residents whether they would choose the bus, to replace the guessed assumption that everyone takes the motorbike.
