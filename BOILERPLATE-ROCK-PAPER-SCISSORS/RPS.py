loz = {}

def player(prev_play, opponent_history=[]):
  global loz

  n = 5

  if prev_play in ["R","P","S"]:
    opponent_history.append(prev_play)

  guess = "R" 

  if len(opponent_history)>n:
    inp = "".join(opponent_history[-n:])

    if "".join(opponent_history[-(n+1):]) in loz.keys():
      loz["".join(opponent_history[-(n+1):])]+=1
    else:
      loz["".join(opponent_history[-(n+1):])]=1

    possible =[inp+"R", inp+"P", inp+"S"]

    for i in possible:
      if not i in loz.keys():
        loz[i] = 0

    predict = max(possible, key=lambda key: loz[key])

    if predict[-1] == "P":
      guess = "S"
    if predict[-1] == "R":
      guess = "P"
    if predict[-1] == "S":
      guess = "R"


  return guess
