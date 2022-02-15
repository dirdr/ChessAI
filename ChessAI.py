import chess
import pygame
import math
import random
import numpy as np
from math import inf as infini
import chess.polyglot
import chess.syzygy
import chess.gaviota
from math import dist
import os



"""
    constant declaration
"""
# put 0.75 in scale if the screen is to big for the computer, it will reduce the size of ther window to 600*600
SCALE = 1
CELL_SIZE = (100*SCALE)

WINDOW_WIDTH = (1100*SCALE)
WINDOW_HEIGHT = (800*SCALE)


BOARD_WIDTH = int(800*SCALE)
BOARD_HEIGHT = int(800*SCALE)

PIECE_SIZE = int(100*SCALE)
FPS = 144
DELTATIME = 1/FPS
# Game state
GAME = 0
MENU = 1
PAUSE = 2
GAME_OVER = 3
# color
DARK_COLOR = "#4b7399"
LIGHT_COLOR = "#eae9d2"
SELECTED_COLOR_DARK = "#2f8ccc"
SELECTED_COLOR_LIGHT = "#75c7e8"

LIGHT_VALID_MOVE_DOT_COLOR = "#d2d1bd"
DARK_VALID_MOVE_DOT_COLOR = "#436789"

pygame.init()
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))



spritesheet = pygame.image.load(os.path.join('assets', 'spritesheet.png'))

bg = pygame.image.load(os.path.join('assets', 'bg2.png')).convert()
white = pygame.image.load(os.path.join('assets', 'whitepawn.png'))
black = pygame.image.load(os.path.join('assets', 'blackpawn.png'))
easy = pygame.image.load(os.path.join('assets', 'easy2.png'))
medium = pygame.image.load(os.path.join('assets', 'medium.png'))
hard = pygame.image.load(os.path.join('assets', 'hard2.png'))
nightmare = pygame.image.load(os.path.join('assets', 'nightmare.png'))
ext = pygame.image.load(os.path.join('assets', 'ext.png')).convert()
draw = pygame.image.load(os.path.join('assets', 'draw.png'))
wwin = pygame.image.load(os.path.join('assets', 'wwin.png'))
bwin = pygame.image.load(os.path.join('assets', 'bwin.png'))
check = pygame.image.load(os.path.join('assets', 'check.png'))
darkcircle = pygame.image.load(os.path.join('assets', 'darkcircle.png'))
lightcircle = pygame.image.load(os.path.join('assets', 'lightcircle.png'))


EXT_WIDTH = (ext.get_width()*SCALE)
EXT_HEIGHT = (ext.get_height()*SCALE)
 
color = [white, black]
difficulty = [easy,medium, hard, nightmare]


HELPER = {'p': 100, 'r': 500, 'n': 320, 'b': 330, 'q': 900, 'k': 0,
        'P': 100, 'R': 500, 'N': 320, 'B': 330, 'Q': 900, 'K': 0}


def get_font(size: int) -> pygame.font.Font:
    return pygame.font.Font(os.path.join('font', 'Futura Heavy Italic font.ttf'), size)


pygame.font.init()
pygame.display.set_caption('ChessAI - ESIEE')

background = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))

clock = pygame.time.Clock()


"""
    select a subsprite (x and y are the top left corner starting position)
    the surface return is in the following format : x + width, y + height
"""
def carve_sprite(image: pygame.Surface, x: int, y: int, width: int, height: int) -> pygame.Surface:
    rect = (x, y, width, height)
    return image.subsurface(rect)


def load_pieces_sprite() -> dict:
    pieces = ['K', 'Q', 'B', 'N', 'R', 'P', 'k', 'q', 'b', 'n', 'r', 'p']
    piece_image = {}
    xs = 0
    ys = 0
    idx = 0
    for i in range(2):
        for j in range(6):
            raw_image = carve_sprite(spritesheet, xs, ys, 200, 200)
            piece_image[pieces[idx]] = pygame.transform.smoothscale(raw_image, (PIECE_SIZE, PIECE_SIZE))
            idx += 1
            xs += 200
        xs = 0
        ys += 200
    return piece_image

piece_image = load_pieces_sprite()

class Game:

    pawn_cell_table = np.array([
            0, 0, 0, 0, 0, 0, 0, 0,
            5, 10, 10, -20, -20, 10, 10, 5,
            5, -5, -10, 0, 0, -10, -5, 5,
            0, 0, 0, 20, 20, 0, 0, 0,
            5, 5, 10, 25, 25, 10, 5, 5,
            10, 10, 20, 30, 30, 20, 10, 10,
            50, 50, 50, 50, 50, 50, 50, 50,
            0, 0, 0, 0, 0, 0, 0, 0
    ])

    knight_cell_table = np.array([
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20, 0, 5, 5, 0, -20, -40,
        -30, 5, 10, 15, 15, 10, 5, -30,
        -30, 0, 15, 20, 20, 15, 0, -30,
        -30, 5, 15, 20, 20, 15, 5, -30,
        -30, 0, 10, 15, 15, 10, 0, -30,
        -40, -20, 0, 0, 0, 0, -20, -40,
        -50, -40, -30, -30, -30, -30, -40, -50
    ])

    bishop_cell_table = np.array([
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10, 5, 0, 0, 0, 0, 5, -10,
        -10, 10, 10, 10, 10, 10, 10, -10,
        -10, 0, 10, 10, 10, 10, 0, -10,
        -10, 5, 5, 10, 10, 5, 5, -10,
        -10, 0, 5, 10, 10, 5, 0, -10,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -20, -10, -10, -10, -10, -10, -10, -20
    ])

    rook_cell_table = np.array([
        0, 0, 0, 5, 5, 0, 0, 0,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        5, 10, 10, 10, 10, 10, 10, 5,
        0, 0, 0, 0, 0, 0, 0, 0
    ])

    queen_cell_table = np.array([
        -20, -10, -10, -5, -5, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 5, 5, 5, 5, 5, 0, -10,
        0, 0, 5, 5, 5, 5, 0, -5,
        -5, 0, 5, 5, 5, 5, 0, -5,
        -10, 0, 5, 5, 5, 5, 0, -10,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -20, -10, -10, -5, -5, -10, -10, -20
    ])

    king_cell_table = np.array([
        20, 30, 10, 0, 0, 10, 30, 20,
        20, 20, 0, 0, 0, 0, 20, 20,
        -10, -20, -20, -20, -20, -20, -20, -10,
        -20, -30, -30, -40, -40, -30, -30, -20,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30
    ])

    king_cell_table_end = np.array([
        -50,-40,-30,-20,-20,-30,-40,-50,
        -30,-20,-10,  0,  0,-10,-20,-30,
        -30,-10, 20, 30, 30, 20,-10,-30,
        -30,-10, 30, 40, 40, 30,-10,-30,
        -30,-10, 30, 40, 40, 30,-10,-30,
        -30,-10, 20, 30, 30, 20,-10,-30,
        -30,-30,  0,  0,  0,  0,-30,-30,
        -50,-30,-30,-30,-30,-30,-30,-50
    ])

    #PIECE_LIST = np.array[chess.PAWN, chess.ROOK, chess.BISHOP, chess.ROOK, chess.QUEEN]

    def __init__(self) -> None:
        self.board = chess.Board()
        self.reader = chess.polyglot.open_reader(os.path.join('OpeningBook', 'Performance.bin'))
        self.evaluation_done = None
        self.evaluated_boards = {}


    def search_syzygy_tablebase(self) -> chess.Move:
        with chess.syzygy.open_tablebase(os.path.join('SyzygyTablebase', 'regular')) as tablebase:
            best_move = chess.Move.null()
            best_score = infini
            for move in self.board.legal_moves:
                self.board.push(move)
                score = -tablebase.probe_dtz(self.board)
                wdl = -tablebase.probe_wdl(self.board) 
                self.board.pop()
                print(f"move : {move}  score {score}  wdl  {wdl}")
                if wdl == 2: # unconditional win
                    if score < best_score:
                        print(f"best_move : {move}  score {score}  wdl  {wdl}")
                        best_score = score
                        best_move = move
            print('\n\n')
            return best_move
        
        
    def search_gaviota_tablebase(self) -> chess.Move:
        with chess.gaviota.open_tablebase('GaviotaTablebase') as tablebase:
            best_move = chess.Move.null()
            best_score = infini

            helper = []
            for move in self.board.legal_moves:
                self.board.push(move)
                wdl = -tablebase.probe_wdl(self.board)
                helper.append(wdl)
                self.board.pop()

            no_winning = True
            all_losing = True
            for probing in helper:
                if probing > 0: no_winning = False
                if probing >= 0: all_losing = False
            
            for move in self.board.legal_moves:
                self.board.push(move)
                score = -tablebase.probe_dtm(self.board)
                wdl = -tablebase.probe_wdl(self.board)
                self.board.pop()
                print(f"move : {move}  score {score}  wdl = {wdl}")
                if wdl == 1:
                    """if the child position is winning, take the minimum dtm"""
                    if score < best_score:   
                        best_score = score
                        best_move = move
                elif wdl == -1:
                    """if the child position is losing, take take the longest dtm"""
                    if all_losing:
                        if score < best_score: 
                            best_score = score
                            best_move = move
                else:
                    """the position is draw return the draw move
                    only if there is no winning position possible"""
                    if no_winning:
                        return move 
            return best_move

    def evaluate(self) -> int:
        """
        Evaluate the given position,
        = 0 the position is equal
        > 0 the position is in favor of the moving side
        < 0 the position is in favor of the other side (the moving side as a disadvantage)
        """

        try : return self.evaluated_boards[self.board.board_fen()]
        except :
            self.evaluation_done += 1

            white_material = self.count_material(chess.WHITE)
            black_material = self.count_material(chess.BLACK)

            single_cell_position_score = self.get_piece_score_sum()

            end_game = self.MopUpEval(self.board.turn)

            evaluation = (white_material - black_material) + single_cell_position_score
            point_of_view = 1 if self.board.turn == chess.WHITE else -1

            self.evaluated_boards[self.board.board_fen()] = evaluation*point_of_view + end_game
            
            # if the evaluation is in favor of white but it's black turn,
            # we need to negate the evaluation because the position will be unfavorable to black
            return evaluation*point_of_view + end_game


    def MopUpEval(self, color: chess.Color) -> int:
        moUpScore = 0
        white = 0
        black = 0
        if self.endgameWeigth():
            friendlyKingSquare = self.board.king(color)
            opponentKingSquare = self.board.king(not color)
            kingdist = chess.square_distance(friendlyKingSquare,opponentKingSquare  ) 
            moUpScore += (7 - kingdist )*50

            return moUpScore + black + white
        return 0


    def endgameWeigth(self) -> bool:
        ff = self.count_material(not self.board.turn)
        return  ff < 500
    
    def get_piece_score_sum(self) -> int:
        """
        get the sum of all individual piece score for a color,
        these individual score are based on the position of the piece
        refer to cell_table
        """
        sum_ = 0
        side = 1
        dict = {chess.PAWN:  Game.pawn_cell_table, chess.ROOK: Game.rook_cell_table, chess.BISHOP : Game.bishop_cell_table, chess.KNIGHT : Game.knight_cell_table , chess.QUEEN : Game.queen_cell_table ,chess.KING : Game.king_cell_table }
        for square in chess.SQUARES:
            side = 1
            sq = square
            if not self.board.piece_at(square) :
                continue
            if self.board.piece_at(square).color == chess.BLACK :
                side = -1
                sq = 63 - square

            if self.board.piece_at(square) == chess.KING and self.endgameWeigth() :
                sum_ += Game.king_cell_table_end[sq] * side
            else :
                sum_ += dict[self.board.piece_at(square).piece_type][sq] * side
        return sum_


    def count_material(self,  color: chess.Color) -> int:
        """
        Count the total material for a given color,
        piece value based on : https://www.chessprogramming.org/Simplified_Evaluation_Function
        """
        sum_ = 0
        sum_ += len(self.board.pieces(chess.PAWN,color)) * 100
        sum_ += len(self.board.pieces(chess.KNIGHT,color)) * 320
        sum_ += len(self.board.pieces(chess.BISHOP,color)) * 330
        sum_ += len(self.board.pieces(chess.ROOK,color)) * 500
        sum_ += len(self.board.pieces(chess.QUEEN,color)) * 900
        return sum_

    def prioritizing_moves(self) -> list:
        """order the list of move, from most potentially interesting to least interesting
            this function is usefull for the alpha-beta prunning algorith to cut off the maximum number of node, 

        Returns:
            list: the ordered legal_moves 
        """
        move_sorted = []
        guesses = []
        for move in self.board.legal_moves:

            guess = 0

            moving_piece = self.board.piece_at(move.from_square)
            target_piece = self.board.piece_at(move.to_square)

            if target_piece != None:
                guess = 10 * HELPER.get(target_piece.symbol()) - HELPER.get(moving_piece.symbol())

            if 'q' in move.uci():
                guess += HELPER.get('Q')

            if self.is_pawn_attacked(moving_piece.color,move.to_square) :
                guess -= HELPER.get(moving_piece.symbol())

            move_sorted += [move]
            guesses += [guess]

        move_sorted = [x for _, x in sorted(zip(guesses, move_sorted), key=lambda pair: pair[0], reverse = True)]
        return move_sorted
    
    def is_pawn_attacked(self, color: chess.Color, square: chess.Square) -> bool:
        """
        Args:
            color (chess.Color): IA color
            square (chess.Square): looking square 

        Returns:
            bool: if a pawn attacked 
        """
        squareset = self.board.attackers(not color, square)
        for square in squareset :
            attacker = self.board.piece_at(square).symbol()
            if attacker == "P" or attacker == "p" : return True
        return False

    def quiet(self, alpha: int, beta: int) -> int:
        """
        continue to search, to limit horizon effect,
        https://www.chessprogramming.org/Quiescence_Search
        """
        score = self.evaluate()
        if score >= beta:
            return beta
        alpha = max(alpha, score)
        for move in self.prioritizing_moves():
            if self.board.is_capture(move):
                self.board.push(move)
                score = -self.quiet(-beta, -alpha)
                self.board.pop()
            if score >= beta:
                return beta
            alpha = max(alpha, score)
        return alpha

    def get_number_of_pieces(self) -> int:
        """
        count the total number of pices on the board,
        for white and black

        Returns:
            int: the number of pieces on the board
        """
        sum_ = 0
        helper = [chess.PAWN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.QUEEN, chess.KING]
        for index in range(len(helper)):
            sum_ += len(self.board.pieces(helper[index], chess.WHITE))
            sum_ += len(self.board.pieces(helper[index], chess.BLACK))
        return sum_
        

    def get_ia_move(self, depth: int) -> chess.Move:
        """
        get the IA move, it can be done by 3 way:
        1 : if the position is still in the opening book, get the weighted_choice
        2 : if the position is not on the opening book, get the move by alpha-beta prunning,
        3 : if there are 4 pieces or less on the board, 
            get the move by looking at gaviota endgame tablebase and the dtm metrics

        Args:
            depth (int): the alpha-beta depth

        Returns:
            chess.Move: the move choose by the IA
        """
        move_from_book = self.get_book_move()
        self.evaluation_done = 0
        if move_from_book is not None:
            return move_from_book
        elif self.get_number_of_pieces() > 4:
            best_move = chess.Move.null()
            best_score = -infini
            alpha = -infini
            beta = infini
            for move in self.prioritizing_moves():
                self.board.push(move)
                score = -self.search(depth, -beta, -alpha)
                print(score)
                if score > best_score:
                    best_score = score
                    best_move = move
                if score > alpha:
                    alpha = score
                self.board.pop()
            return best_move
        else:
            return self.search_gaviota_tablebase()

    def search(self, depth: int, alpha: int, beta: int) -> int:
        """
        Negamax implemented Alpha-Beta prunning algorithm
        """

        if depth == 0: # max depth reached
            return self.quiet(alpha, beta)

        if self.board.is_checkmate() : # checkmate
                return -200000

        
        if self.board.is_stalemate() or self.board.is_repetition(3) or self.board.is_seventyfive_moves() or self.board.has_insufficient_material(not self.board.turn): # draw
            return 0 

        for move in self.prioritizing_moves():
            self.board.push(move)
            score = -self.search(depth - 1, -beta, -alpha)
            self.board.pop()
            if score >= beta:
                return beta
            alpha = max(alpha, score)
        return alpha

        
    def get_book_move(self) -> chess.Move:
        """
            try to get a book-move from the polyglot book,
            if the position is not found, search the move with Alpha-Beta prunning
        """
        try:
            entry = self.reader.weighted_choice(self.board)
        except IndexError:
            return None
        move = entry.move
        return move


    def get_random_move(self, color: chess.Color) -> chess.Move:
        """
        choose a random move for the color arg
        """
        all_legal_move = self.get_all_legal_move(color)
        move = random.choice(all_legal_move)
        return move


    def get_all_legal_move(self, color: chess.Color) -> list:
        """
        return a list off all the legal move for color arg,
        """
        returnable = []
        for cell in range(64):
            if self.board.piece_at(cell) != None and self.board.piece_at(cell).color == color:
                legal_move = self.get_legal_move(cell)
                for move in legal_move:
                    returnable.append(move)
        return returnable


    def get_legal_move(self, starting_cell: chess.Square) -> list:
        """
        return a list of all the legal move for the piece on the starting_cell arg
        """
        returnable = []
        for fcell in chess.SQUARE_NAMES:
            move = chess.Move(starting_cell, chess.parse_square(fcell))
            if move in self.board.legal_moves:
                returnable.append(move)
        return np.array(returnable)


    def get_legal_cell(self, legal_move: list) -> np.array:
        """
        Take a list of legal move as arg and return the corresponding cell as a list
        """
        returnable = []
        for move in legal_move:
            returnable.append(move.to_square)
        return np.array(returnable)

    
    def check_if_legal_move(self, move: chess.Move) -> bool:
        """
        return true if the move is legal, else false
        """
        return move in self.board.legal_moves


    def do_move(self, move: chess.Move, user_color: chess.Color) -> None:
        """
        check if you can promote the move, if you can't, 
        check if the move is legal.
        if the move is legal, push it
        """
        if self.board.turn == user_color:
            if chess.Move.from_uci(move.uci()+"q") in self.board.legal_moves:
                self.board.push(chess.Move.from_uci(move.uci()+"q"))
                return    
        if self.check_if_legal_move(move): # the move is legal, we do it
            self.board.push(move)

    def reset_board(self) -> None:
        """
        reset the game board
        """
        self.board = chess.Board()



class Renderer:

    def __init__(self, game: Game) -> None:
        self.game = game
        self.user_color = None
        self.piece_position = self.reset_piece_position()
        self.selected_cell = -1
        self.moving_piece_cell = -1
        self.pointer_pos = (None, None)
        self.anim_timer = 0
        self.user_dragging = False
        self.history = []

    def set_user_color(self, user_color: chess.Color) -> None:
        self.user_color = user_color


    def animate_move(self, move: chess.Move) -> None:
        """
        animate the arg move on the board, 
        this speed off the anim is set by the variable 'anim_duration'
        """
        anim_duration: float = 0.10

        from_c = move.from_square
        to_c = move.to_square

        from_co = get_cell_coordinate_from_index(from_c, self.user_color)
        to_co = get_cell_coordinate_from_index(to_c, self.user_color)

        xp = [0, 1]
        fpx = [from_co[0], to_co[0]]
        fpy = [from_co[1], to_co[1]]

        
        while self.anim_timer <= anim_duration:

            self.anim_timer += DELTATIME
            alpha = self.anim_timer/anim_duration
            rx = np.interp(alpha, xp, fpx)
            ry = np.interp(alpha, xp, fpy)
            self.piece_position[from_c] = (rx, ry)
            self.interruption_draw(False)

        # animation finished, reset the animation timer and push the move
        self.anim_timer = 0
        self.game.do_move(move, self.user_color)


    def interruption_draw(self, draw_extra: bool) -> None:
        """
        This function is not meant to the main draw function
        it can be used to animate piece for exemple, when the renderer need to update the display
        """
        self.render_game(draw_extra)
        self.render_blank_side()
        window.blit(background, (0, 0))
        pygame.display.update()


    def render_game(self, draw_extra) -> None:
        """
        Main render function
        """
        self.render_board()
        if draw_extra:

            self.render_legal_move()
            self.render_selected_cell()
            self.render_check()

        self.render_piece()
        
        background.blit(ext, (BOARD_WIDTH, 0 ) )

    def render_blank_side(self) -> None:
        background.blit(ext, (WINDOW_WIDTH-EXT_WIDTH, 0))
        
        BLUE = (74, 114, 152)
        
        eval_text_render = get_font(30).render("EVALUATED", True, BLUE)
        eval_text_render2 = get_font(30).render("MOVES", True, BLUE)
        
        move_text_render = get_font(30).render("CHOSEN", True, BLUE)
        move_text_render2 = get_font(30).render("MOVE", True, BLUE)
        
        background.blit(eval_text_render, (BOARD_WIDTH + (EXT_WIDTH-eval_text_render.get_width())/2, 60))
        background.blit(eval_text_render2, (BOARD_WIDTH + (EXT_WIDTH-eval_text_render2.get_width())/2, 100))
        
        background.blit(move_text_render, ( BOARD_WIDTH + (EXT_WIDTH-move_text_render.get_width() )/2, 240))
        background.blit(move_text_render2, ( BOARD_WIDTH + (EXT_WIDTH-move_text_render2.get_width() )/2, 280)) 
        

    def render_content_side(self, evaluated_position: int, move: chess.Move) -> None:

        BLUE = (74, 114, 152)
        LIGHTBLUE = (124, 164, 202)
        DARKBLUE = (24, 64, 102)
        
        if move == chess.Move.null(): move_str = ""
        else: move_str = move.uci()

        eval_text_render = get_font(30).render("EVALUATED", True, BLUE)
        eval_text_render2 = get_font(30).render("MOVES", True, BLUE)

        
        if self.game.get_number_of_pieces() <= 4:
            evaluation_text = "Gaviota"
            eval_content_text_render = get_font(40).render(evaluation_text, True, LIGHTBLUE)
        elif evaluated_position == 0:
            evaluation_text = "Book"
            eval_content_text_render = get_font(40).render(evaluation_text, True, LIGHTBLUE)
        else:
            evaluation_text = str(evaluated_position)
            eval_content_text_render = get_font(40).render(evaluation_text, True, DARKBLUE)
            

        move_text_render = get_font(30).render("CHOSEN", True, BLUE)
        move_text_render2 = get_font(30).render("MOVE", True, BLUE)
        
        move_content_text_render = get_font(40).render(move_str, True, DARKBLUE)

        background.blit(eval_text_render, (BOARD_WIDTH + (EXT_WIDTH-eval_text_render.get_width())/2, 60) )
        background.blit(eval_text_render2, (BOARD_WIDTH + (EXT_WIDTH-eval_text_render2.get_width())/2, 100) )

        if evaluated_position is not None:
            background.blit(eval_content_text_render, (BOARD_WIDTH+ (EXT_WIDTH-eval_content_text_render.get_width())/2, 160) )

        background.blit(move_text_render, ( BOARD_WIDTH + (EXT_WIDTH-move_text_render.get_width() )/2, 240))
        background.blit(move_text_render2, ( BOARD_WIDTH + (EXT_WIDTH-move_text_render2.get_width() )/2, 280))

        background.blit(move_content_text_render, (BOARD_WIDTH + (EXT_WIDTH-move_content_text_render.get_width())/2, 340) )

        
    def render_game_over(self) -> None:
        """
        render the game_over screen into the background
        """
        outcome = self.game.board.outcome()
        termination = outcome.termination
        winner = outcome.winner
        if winner == chess.WHITE : w = wwin
        elif winner == chess.BLACK : w = bwin
        else : w = draw
        self.render_board()
        self.reset_piece_position()
        self.render_piece()
        background.blit(w,(0,0))


    def render_check(self) -> None:
        if self.game.board.is_check():
            color = self.game.board.turn
            king_index = self.game.board.pieces(chess.KING, color)
            index = list(king_index)[0]
            king_coordinate = get_cell_coordinate_from_index(index, self.user_color)

            if self.game.board.turn != self.user_color: 
                background.blit(check, king_coordinate)
            elif self.user_dragging == False:
                background.blit(check, king_coordinate)


    def render_menu(self, depth: int) -> None:
        """
            render the menu into the background
        """
        if self.user_color == chess.WHITE : x = 0
        else : x = 1
        background.blit(bg,(0,0))
        background.blit(color[x],(207,312))
        background.blit(difficulty[depth-1],(665,312))


    def render_board(self) -> None:
        """
        render an empty board into the background
        """
        for index in range(64):
            x, y = get_cell_coordinate_from_index(index, self.user_color)
            color = LIGHT_COLOR if is_light(index) else DARK_COLOR
            pygame.draw.rect(background, color = color, rect = pygame.Rect(
                                                                            x,
                                                                            y,
                                                                            CELL_SIZE,
                                                                            CELL_SIZE
                                                                        ))
            
            
    def render_last_move(self, last_move: chess.Move) -> None:
        if last_move is None: return
        last_move_starting = last_move.from_square
        last_move_ending = last_move.to_square
        
        lms = get_cell_coordinate_from_index(last_move_starting, self.user_color)
        lme = get_cell_coordinate_from_index(last_move_ending, self.user_color)

        lms_color = SELECTED_COLOR_LIGHT if is_light(last_move_starting) else SELECTED_COLOR_DARK
        lme_color = SELECTED_COLOR_LIGHT if is_light(last_move_ending) else SELECTED_COLOR_DARK
        
        pygame.draw.rect(background, color = lms_color, rect = pygame.Rect(
                                                                        lms[0],
                                                                        lms[1],
                                                                        CELL_SIZE,
                                                                        CELL_SIZE
                                                                    ))
        
        pygame.draw.rect(background, color = lme_color, rect = pygame.Rect(
                                                                        lme[0],
                                                                        lme[1],
                                                                        CELL_SIZE,
                                                                        CELL_SIZE
                                                                    ))
        
        
        

    def render_selected_cell(self) -> None:
        """
        render the selected square
        """
        if self.selected_cell != -1 and self.game.board.piece_at(self.selected_cell) != None:
            x, y = get_cell_coordinate_from_index(self.selected_cell, self.user_color)
            color = SELECTED_COLOR_LIGHT if is_light(self.selected_cell) else SELECTED_COLOR_DARK
            pygame.draw.rect(background, color = color, rect = pygame.Rect(
                                                                        x,
                                                                        y,
                                                                        CELL_SIZE,
                                                                        CELL_SIZE
                                                                    ))


    def render_legal_move(self) -> None:
        """
        a dot will be render on the cell considered legal.
        a cell is legal if it is the target square of a legal move
        """
        if self.selected_cell != None:
        
            legal_move = self.game.get_legal_move(self.selected_cell)
            for move in legal_move:

                cell = move.to_square
                coordinate = get_cell_coordinate_from_index(cell, self.user_color)
               
                if self.game.board.is_capture(move):
                    image = lightcircle if is_light(cell) else darkcircle
                    background.blit(image, coordinate)
                else:
                    x, y = coordinate[0]+CELL_SIZE/2, coordinate[1]+CELL_SIZE/2
                    color = LIGHT_VALID_MOVE_DOT_COLOR if is_light(cell) else DARK_VALID_MOVE_DOT_COLOR
                    pygame.draw.circle(background, center = (x, y), radius = CELL_SIZE/6, color = color)

    
    def render_piece(self) -> None:
        """
        Render all the pieces into the board,
        render the moving piece into the board
        """
        for index in range(64):
            piece = self.game.board.piece_at(index)
            if piece != None and index != self.moving_piece_cell:
                symbol = piece.symbol()
                x, y = self.piece_position[index]
                background.blit(piece_image.get(symbol), pygame.Rect(
                                                                    x,
                                                                    y,
                                                                    PIECE_SIZE,
                                                                    PIECE_SIZE)
                                                                )
        if self.user_dragging and self.moving_piece_cell != -1:
            x, y = self.pointer_pos[0]-CELL_SIZE/2, self.pointer_pos[1]-CELL_SIZE/2
            piece = self.game.board.piece_at(self.moving_piece_cell)
            symbol = piece.symbol()
            background.blit(piece_image.get(symbol), pygame.Rect(
                                                                x,
                                                                y,
                                                                PIECE_SIZE,
                                                                PIECE_SIZE)
                                                            )


    def reset_piece_position(self) -> None:
        self.piece_position = [get_cell_coordinate_from_index(index, self.user_color) for index in range(64)]



class GameManager:

    def __init__(self):
        self.game = Game()
        self.user_color = chess.WHITE
        self.game_state = MENU
        self.depth = 1 # the menu is loaded with the difficulty : easy
        self.renderer = Renderer(self.game)
        self.last_ia_move = chess.Move.null()
        self.last_move = None


    def wait(self, time: float) -> None:
        """
        Args:
            time (float): number of time to wait
        """
        temp = 0
        while (temp < time):
            temp += DELTATIME
        return


    def handle_mouse(self, event: pygame.event) -> None:

        """
        handle the user mouse
        into all the different possible screen (game, menu, game_over)
        """

        self.renderer.pointer_pos = pygame.mouse.get_pos()
        self.renderer.reset_piece_position()

        # if the left click is pressed
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:

            if self.game_state == GAME:

                pointed_index = get_index_from_coordinate(self.renderer.pointer_pos, self.user_color)
                if self.game.board.piece_at(pointed_index) != None:
                    self.renderer.user_dragging = True
                    self.renderer.selected_cell = pointed_index
                    self.renderer.moving_piece_cell = pointed_index
                    self.renderer.piece_position[self.renderer.moving_piece_cell] = (-100, -100)


        # if the left click is unpressed
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            
            x, y = self.renderer.pointer_pos

            if self.game_state == GAME:

                drop_index = get_index_from_coordinate(self.renderer.pointer_pos, self.user_color)
                if self.renderer.moving_piece_cell == drop_index:
                    move = chess.Move.null
                else:
                    move = chess.Move(self.renderer.moving_piece_cell, drop_index)
                    self.last_move = move
                    self.game.do_move(move, self.user_color)

                self.renderer.moving_piece_cell = -1
                self.renderer.user_dragging = False

            elif self.game_state == MENU:

                # click on the play button
                if 408<x<708 and 602<y<732 : self.game_state = GAME

                # changing the color played by the user
                if 224<x<428 and 328<y<535 :
                    if self.user_color == chess.WHITE : self.user_color = chess.BLACK
                    else : self.user_color = chess.WHITE

                # changing the difficulty
                if 678<x<886 and 328<y<535 :
                    self.depth += 1
                    if self.depth > 4 : self.depth = 1

            if 880<x<1024 and 474<y<530:
                self.reset()


    def reset(self) -> None:
        """
        reset the board, user color and difficulty,
        change the game state to MENU
        """
        self.game.reset_board()
        self.depth = 1
        self.user_color = chess.WHITE
        self.game_state = MENU
        

    def handle_outcome(self) -> None:
        """
        change the game state according to the game outcome
        """
        outcome = self.game.board.outcome()
        if outcome != None:
            self.game_state = GAME_OVER
            
    def gm_interruption_darw(self) -> None:
        self.renderer.render_board()
        self.renderer.render_selected_cell()
        self.renderer.render_last_move(self.last_move)
        self.renderer.render_legal_move()
        self.renderer.render_check()
        self.renderer.render_piece() # reset the piece position on the board and draw it before the ai thinking
        background.blit(ext, (BOARD_WIDTH, 0 ) )
        window.blit(background, (0, 0))
        pygame.display.update()

    def update(self) -> None:
        """
        main update function, called every frame,
        handle user moving piece, finding the ai move, rendering
        """
        
        """core part"""
        self.renderer.set_user_color(self.user_color)
        self.handle_outcome() 

        if self.game_state == GAME: # game

            if self.game.board.turn != self.renderer.user_color:
                
                self.gm_interruption_darw()
                
                ia_move = self.game.get_ia_move(self.depth)
                
                self.wait(10) # wait a little bit before rendering the bot doing the move
                
                self.last_ia_move = ia_move
                self.last_move = ia_move
                self.renderer.animate_move(ia_move)


        """render part"""
        if self.game_state == MENU:
            self.renderer.render_menu(self.depth)
        elif self.game_state == GAME:
            
            self.renderer.render_board()
            self.renderer.render_selected_cell()
            self.renderer.render_last_move(self.last_move)
            self.renderer.render_legal_move()
            self.renderer.render_check()
            self.renderer.render_piece()
            background.blit(ext, (BOARD_WIDTH, 0 ) )
            
            self.renderer.render_content_side(self.game.evaluation_done, self.last_ia_move)
            
        elif self.game_state == GAME_OVER:
            self.renderer.render_game_over()

        window.blit(background, (0, 0))
        pygame.display.update()



def is_light(cell_index: int) -> bool:
    """
    return if the cell arg is light or dark
    """
    file = chess.square_file(cell_index)
    rank = chess.square_rank(cell_index)
    is_light = (file + rank) % 2 != 0
    return is_light


def get_cell_coordinate_from_index(index: int, user_color: chess.Color) -> tuple:
    """
    transform the cell index arg into a pygame_coordinate,
    the user_color arg is used to determine which side to draw bottom 
    (user_color == chess.WHITE -> white bottom)
    (user_color == chess.BLACK -> black bottom)
    """

    if user_color == chess.WHITE: 
        rank = chess.square_rank(index)
        file = chess.square_file(index)
    else:
        rank = 7 - chess.square_rank(index)
        file = 7 - chess.square_file(index)

    return  (file * PIECE_SIZE, BOARD_HEIGHT - rank*PIECE_SIZE - PIECE_SIZE)


def get_rf_from_coordinate(pos: tuple) -> tuple:
    """
    get the rank and file as a tuple, from the position (x, y coordinate )
    """
    x, y = pos[0], pos[1]
    file, rank = x/CELL_SIZE, (BOARD_HEIGHT-y)/CELL_SIZE
    return (math.floor(rank), math.floor(file))


def get_index_from_coordinate(pos: tuple, user_color: chess.Color) -> int:
    """
    return the cell index of the arg pos
    """
    x, y = pos[0], pos[1]
    r, f = get_rf_from_coordinate((x, y))
    cell = chess.square(file_index = f, rank_index = r)
    if user_color == chess.WHITE: return cell
    return 63 - cell

def main() -> None:
    gm = GameManager()
    # Main loop
    is_running = True
    while is_running:

        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_running = False
            gm.handle_mouse(event)
        gm.update()

if __name__ == '__main__':
    main()