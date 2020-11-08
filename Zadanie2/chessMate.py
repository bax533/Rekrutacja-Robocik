import copy

boardLetters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
knightMoves = frozenset([(-1, -2), (1, -2), (2, -1), (2,1), (1, 2), (-1, 2), (-2, 1), (-2,-1)])
bishopMoves = frozenset([(-1, -1), (1, -1), (1, 1), (-1, 1)])
rookMoves = frozenset([(-1,0), (1, 0), (0, -1), (0, 1)])
kingMoves = frozenset([(-1, -1), (1, -1), (1, 1), (-1, 1), (-1,0), (1, 0), (0, -1), (0, 1)])

def OnBoard(pos):
	return pos[0] >= 0 and pos[0] <= 7 and pos[1] >= 0 and pos[1] <= 7

class Pawn:
	def __init__(self, name, pos):
		self.name = name
		self.pos = pos

	def __eq__(self, o):
		return self.name == o.name and self.pos == o.pos

	def __hash__(self):
		return hash(self.name) ^ hash(self.pos)

class State:
	def __init__(self, whiteFigures, blackFigures, board):
		self.whiteFigures = whiteFigures
		self.blackFigures = blackFigures
		self.board = board
		self.bKingPos = self.GetBlackKingPos()

	def Actions(self, color):
		ret = set()
		if(color == 'white'):
			for pawn in self.whiteFigures:
				if(pawn.name[1] == 'p'): # pionek

					newPos = (pawn.pos[0], pawn.pos[1]-1)
					if(OnBoard(newPos) and self.board[newPos] == '--'): # 1 góra
						ret.add((pawn.pos, newPos))

					if(pawn.pos[1] == 6):
						newPos = (pawn.pos[0], pawn.pos[1]-2)	# 2 góra jak na starcie
						if(self.board[newPos] == '--'):
							ret.add((pawn.pos, newPos))

					newPos = (pawn.pos[0]+1, pawn.pos[1]-1)
					if(OnBoard(newPos) and self.board[newPos][0] == 'b'): #bicie w prawo
						ret.add((pawn.pos, newPos))

					newPos = (pawn.pos[0]-1, pawn.pos[1]-1)
					if(OnBoard(newPos) and self.board[newPos][0] == 'b'): #bicie w lewo
						ret.add((pawn.pos, newPos))

				if(pawn.name[1] == 'r'):
					for mov in rookMoves:
						newPos = copy.deepcopy(pawn.pos)
						inTheWay = False
						newPos = (newPos[0] + mov[0], newPos[1] + mov[1])
						
						while(OnBoard(newPos)):
							if(self.board[newPos] != '--'):
								inTheWay = True
								break

							ret.add((pawn.pos, newPos))

							newPos = (newPos[0] + mov[0], newPos[1] + mov[1])
							
						if(inTheWay and self.board[newPos][0] == 'b'):
							ret.add((pawn.pos, newPos))

				if(pawn.name[1] == 'k'):
					for mov in knightMoves:
						newPos = (pawn.pos[0] + mov[0], pawn.pos[1] + mov[1])
						if(OnBoard(newPos) and self.board[newPos][1] != 'w'):
							ret.add((pawn.pos, newPos))

				if(pawn.name[1] == 'b'):
					for mov in bishopMoves:
						newPos = copy.deepcopy(pawn.pos)
						inTheWay = False
						newPos = (newPos[0] + mov[0], newPos[1] + mov[1])
						
						while(OnBoard(newPos)):
							if(self.board[newPos] != '--'):
								inTheWay = True
								break

							ret.add((pawn.pos, newPos))

							newPos = (newPos[0] + mov[0], newPos[1] + mov[1])
							
						if(inTheWay and self.board[newPos][0] == 'b'):
							ret.add((pawn.pos, newPos))

				if(pawn.name[1] == 'q'):
					for mov in kingMoves:
						newPos = copy.deepcopy(pawn.pos)
						inTheWay = False
						newPos = (newPos[0] + mov[0], newPos[1] + mov[1])
						
						while(OnBoard(newPos)):
							if(self.board[newPos] != '--'):
								inTheWay = True
								break

							ret.add((pawn.pos, newPos))

							newPos = (newPos[0] + mov[0], newPos[1] + mov[1])
							
						if(inTheWay and self.board[newPos][0] == 'b'):
							ret.add((pawn.pos, newPos))
		else:
			for pawn in self.blackFigures:
				if(pawn.name[1] == 'p'): # pionek

					newPos = (pawn.pos[0], pawn.pos[1]+1)
					if(OnBoard(newPos) and self.board[newPos] == '--'): # 1 góra
						ret.add((pawn.pos, newPos))

					if(pawn.pos[1] == 1):
						newPos = (pawn.pos[0], pawn.pos[1]+2)	# 2 góra jak na starcie
						if(self.board[newPos] == '--'):
							ret.add((pawn.pos, newPos))

					newPos = (pawn.pos[0]+1, pawn.pos[1]+1)
					if(OnBoard(newPos) and self.board[newPos][0] == 'w'): #bicie w prawo
						ret.add((pawn.pos, newPos))

					newPos = (pawn.pos[0]-1, pawn.pos[1]+1)
					if(OnBoard(newPos) and self.board[newPos][0] == 'w'): #bicie w lewo
						ret.add((pawn.pos, newPos))

				if(pawn.name[1] == 'r'):
					for mov in rookMoves:
						newPos = copy.deepcopy(pawn.pos)
						inTheWay = False
						newPos = (newPos[0] + mov[0], newPos[1] + mov[1])
						
						while(OnBoard(newPos)):
							if(self.board[newPos] != '--'):
								inTheWay = True
								break

							ret.add((pawn.pos, newPos))

							newPos = (newPos[0] + mov[0], newPos[1] + mov[1])
							
						if(inTheWay and self.board[newPos][0] == 'w'):
							ret.add((pawn.pos, newPos))

				if(pawn.name[1] == 'k'):
					for mov in knightMoves:
						newPos = (pawn.pos[0] + mov[0], pawn.pos[1] + mov[1])
						if(OnBoard(newPos) and self.board[newPos][1] != 'b'):
							ret.add((pawn.pos, newPos))

				if(pawn.name[1] == 'b'):
					for mov in bishopMoves:
						newPos = copy.deepcopy(pawn.pos)
						inTheWay = False
						newPos = (newPos[0] + mov[0], newPos[1] + mov[1])
						
						while(OnBoard(newPos)):
							if(self.board[newPos] != '--'):
								inTheWay = True
								break

							ret.add((pawn.pos, newPos))

							newPos = (newPos[0] + mov[0], newPos[1] + mov[1])
							
						if(inTheWay and self.board[newPos][0] == 'w'):
							ret.add((pawn.pos, newPos))

				if(pawn.name[1] == 'q'):
					for mov in kingMoves:
						newPos = copy.deepcopy(pawn.pos)
						inTheWay = False
						newPos = (newPos[0] + mov[0], newPos[1] + mov[1])
						
						while(OnBoard(newPos)):
							if(self.board[newPos] != '--'):
								inTheWay = True
								break

							ret.add((pawn.pos, newPos))

							newPos = (newPos[0] + mov[0], newPos[1] + mov[1])
							
						if(inTheWay and self.board[newPos][0] == 'w'):
							ret.add((pawn.pos, newPos))
		return ret


	def Succ(self, a):
		beforePos = a[0]
		afterPos = a[1]

		newBoard = copy.deepcopy(self.board)
		newWhiteFigures = copy.deepcopy(self.whiteFigures)
		newBlackFigures = copy.deepcopy(self.blackFigures)

		#print(a)
		#self.dbg()

		befPawn = self.GetPawnFromPos(beforePos)
		afterPawn = Pawn(befPawn.name, afterPos)

		if(befPawn.name[0]=='w'):
			newWhiteFigures.remove(befPawn)
			newBoard[befPawn.pos] = '--'

			newWhiteFigures.add(afterPawn)
			newBoard[afterPawn.pos] = afterPawn.name

			if(self.board[afterPawn.pos][0] == 'b'):
				newBlackFigures.remove(Pawn(self.board[afterPawn.pos], afterPawn.pos))

			return State(newWhiteFigures, newBlackFigures, newBoard)
		else:
			newBlackFigures.remove(befPawn)
			newBoard[befPawn.pos] = '--'

			newBlackFigures.add(afterPawn)
			newBoard[afterPawn.pos] = afterPawn.name

			if(self.board[afterPawn.pos][0] == 'w'):
				newWhiteFigures.remove(Pawn(self.board[afterPawn.pos], afterPawn.pos))

			return State(newWhiteFigures, newBlackFigures, newBoard)


	def GetPawnFromPos(self, pos):
		for pawn in self.whiteFigures:
			if(pawn.pos == pos):
				return pawn
		for pawn in self.blackFigures:
			if(pawn.pos == pos):
				return pawn

	def GetBlackKingPos(self):
		for pawn in self.blackFigures:
			if(pawn.name == 'bW'):
				return pawn.pos

	def WhiteMate(self):
		for mov in kingMoves:
			newPos = (self.bKingPos[0] + mov[0], self.bKingPos[1] + mov[1])
			inTheWay = False
			while(OnBoard(newPos)):
				if(self.board[newPos] != '--'):
					inTheWay = True
					break
				newPos = (newPos[0] + mov[0], newPos[1] + mov[1])
				

			if(inTheWay):
				if(self.board[newPos][0] == 'b'):
					continue
				else:
					if(self.board[newPos][1] == 'q'):
						return True
					elif(self.board[newPos][1] == 'r'):
						if(mov in rookMoves):
							return True
					elif(self.board[newPos][1] == 'b'):
						if(mov in bishopMoves):
							return True

		for mov in knightMoves:
			newPos = (self.bKingPos[0] + mov[0], self.bKingPos[1] + mov[1])
			if(OnBoard(newPos) and self.board[newPos] == 'wk'):
				return True

		newPos = (self.bKingPos[0] + 1, self.bKingPos[1] + 1)
		if(OnBoard(newPos) and self.board[newPos] == 'wp'):
			return True

		newPos = (self.bKingPos[0] - 1, self.bKingPos[1] + 1)
		if(OnBoard(newPos) and self.board[newPos] == 'wp'):
			return True

		return False

	def WhiteCheckmate(self):
		if(not self.WhiteMate()):
			return False

		for mov in kingMoves:
			newPos = (self.bKingPos[0] + mov[0], self.bKingPos[1] + mov[1])
			if(not OnBoard(newPos)):
				continue

			newWhiteFigures = copy.deepcopy(self.whiteFigures)
			newBlackFigures = copy.deepcopy(self.blackFigures)
			newBoard = copy.deepcopy(self.board)

			newBlackFigures.remove(Pawn('bW', self.bKingPos))
			newBoard[self.bKingPos] = '--'

			if(newBoard[newPos][0] == 'b'):
				continue
			elif(newBoard[newPos][0] == 'w'):
				newWhiteFigures.remove(Pawn(newBoard[newPos], newPos))

			newBlackFigures.add(Pawn('bW', newPos))
			newBoard[newPos] = 'bW'

			newState = State(newWhiteFigures, newBlackFigures, newBoard)
			if(not newState.WhiteMate()):
				#print("tutaj szachu nie ma:")
				#print(newState.bKingPos)
				#newState.dbg()
				return False

		for a in self.Actions('black'):
			newState = self.Succ(a)
			if(not newState.WhiteMate()):
				#print(a)
				return False

		return True

	def dbg(self):
		for y in range(8):
			for x in range(8):
				print(self.board[(x, y)], end = ' ')
			print("") 

		for p in self.blackFigures:
			print(p.name, p.pos)

def PrepareBoard(board):
	retWhiteFigures = set()
	retBlackFigures = set()

	for y in range(8):
		for x in range(8):
			if board[y][x][0] == 'w':
				retWhiteFigures.add(Pawn(board[y][x], (x, y)))
			elif board[y][x][0] == 'b':
				retBlackFigures.add(Pawn(board[y][x], (x, y)))

	return retWhiteFigures, retBlackFigures

def DictFromBoard(board):
	retDict = {}
	for y in range(8):
		for x in range(8):
			retDict[(x, y)] = board[y][x]
	return retDict

def ConvertToChessCords(moves):
	ret = []
	for a in moves:
		bef = a[0]
		after = a[1]
		newBef = (boardLetters[bef[0]] + str(8-bef[1]))
		newAfter = (boardLetters[after[0]] + str(8-after[1]))
		ret.append((newBef, newAfter))
	return ret

f = open("board.txt", "r")
inp = f.read().split(']')

board = []
for x in inp:
	if(len(x) < 8):
		continue
	thisLine = []
	for a in x.split(','):
		while(a[0] == '[' or a[0] == "'" or a[0] == ' ' or a[0] == '\n'):
			a = a[1:]
		thisLine.append(a[:len(a)-1])

	board.append(thisLine)


boardDict = DictFromBoard(board)
whiteFigures, blackFigures = PrepareBoard(board)


state = State(whiteFigures, blackFigures, boardDict)
mateMoves = []

for a in state.Actions('white'):
	newState = state.Succ(a)
	if(newState.WhiteCheckmate()):
		mateMoves.append(a)

print(ConvertToChessCords(mateMoves))

