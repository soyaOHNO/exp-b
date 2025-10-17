# 情報工学実験B 人工知能実験 関数ファイル
#
#   Consoleを開き，%run ebai.py を実行すると，構文エラーの有無を確認できます．
#   人工知能実験では，このファイルの書き換えは，基本しないようにお願いします．
#   （この中の関数に対する工夫があっても，評価の対象外です．）
#
################################################################################
import os
from datetime import datetime
import sys
from dataclasses import dataclass

import IPython.display
from IPython.display import display, Audio, Markdown

try:
    import ipywidgets
except:
    sys.stderr.print('Info: ipywidgets is not installed.')
    pass

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import networkx as netx

matplotlib.rcParams["savefig.bbox"] = 'tight'

################################################################################
__version__ = '2025.1'
__author__ = 'Sunao HARA'

################################################################################
def ebai_info() -> None:
    """
    本ファイルのバージョン情報を表示する．
    """
    print(__file__)
    print(f'> Revision {__version__}')
    print(f'> 更新日時：{datetime.fromtimestamp(os.path.getmtime(__file__))}')

    # ツールのバージョン依存の問題が起こることもあります．(Pythonでは特に顕著)
    # 重要なツールについては「動作確認の環境」として，ツール名だけでなくバージョン名も書く習慣をつけましょう．
    #
    print('>')
    print(f'> Python     : {sys.version}')
    print(f'> NumPy      : {np.__version__}')
    print(f'> Matplotlib : {matplotlib.__version__}')
    print(f'> NetworkX   : {netx.__version__}')

################################################################################
# Original source code
# https://stackoverflow.com/questions/22785849/drawing-multiple-edges-between-two-nodes-with-networkx
def _custom_draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=None,
    label_pos=0.5,
    font_size=10,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=None,
    bbox=None,
    horizontalalignment="center",
    verticalalignment="center",
    ax=None,
    rotate=True,
    clip_on=True,
    rad=0
) -> dict:
    """Draw edge labels.

    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    edge_labels : dictionary (default={})
        Edge labels in a dictionary of labels keyed by edge two-tuple.
        Only labels for the keys in the dictionary are drawn.

    label_pos : float (default=0.5)
        Position of edge label along edge (0=head, 0.5=center, 1=tail)

    font_size : int (default=10)
        Font size for text labels

    font_color : string (default='k' black)
        Font color string

    font_weight : string (default='normal')
        Font weight

    font_family : string (default='sans-serif')
        Font family

    alpha : float or None (default=None)
        The text transparency

    bbox : Matplotlib bbox, optional
        Specify text box properties (e.g. shape, color etc.) for edge labels.
        Default is {boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)}.

    horizontalalignment : string (default='center')
        Horizontal alignment {'center', 'right', 'left'}

    verticalalignment : string (default='center')
        Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    rotate : bool (deafult=True)
        Rotate edge labels to lie parallel to edges

    clip_on : bool (default=True)
        Turn on clipping of edge labels at axis boundaries

    Returns
    -------
    dict
        `dict` of labels keyed by edge

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edge_labels = nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw
    draw_networkx
    draw_networkx_nodes
    draw_networkx_edges
    draw_networkx_labels
    """

    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in G.edges(data=True)}
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )
        pos_1 = ax.transData.transform(np.array(pos[n1]))
        pos_2 = ax.transData.transform(np.array(pos[n2]))
        linear_mid = 0.5*pos_1 + 0.5*pos_2
        d_pos = pos_2 - pos_1
        rotation_matrix = np.array([(0, 1), (-1, 0)])
        ctrl_1 = linear_mid + rad*rotation_matrix@d_pos
        ctrl_mid_1 = 0.5*pos_1 + 0.5*ctrl_1
        ctrl_mid_2 = 0.5*pos_2 + 0.5*ctrl_1
        bezier_mid = 0.5*ctrl_mid_1 + 0.5*ctrl_mid_2
        (x, y) = ax.transData.inverted().transform(bezier_mid)

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            # label = str(label)  # this makes "1" and 1 labeled the same
            label = "{:.1f}".format(label)

        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=1,
            clip_on=clip_on,
        )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items


################################################################################
#
# draw the Graph
#
def draw_graph_structure(
    graph_A, graph_pos=None, graph_target=None,
    *,
    highlight_path=None, open_list=None, closed_list=None,
    show_weight=False,
    ax=None,
    figsize=(6, 4),
    nscale=1.0,
    need_return=False
):
    """
    Parameters
    ----------
    graph_A :
        隣接行列 (Adjacency Matrix) または 隣接リスト (Adjacency List)

        - `[[w11, w12, ...], [w21, w22, ...], ...]`: 隣接行列
        - `[(src1, dst1, weight1), (src2, dst2, weight2), ...]`
        - `{src1: [(dst1, weight1), (dst2, weight2), ...], src2: ...}`
        - `{[(dst1, weight1), (dst2, weight2), ...], ...}` ※先頭のリストからsrc=0, src=1, ...

    graph_pos : list of dict
        各ノードを表示すべき座標

    graph_target : int
        探索を終了するノードのID

    highlight_path : list of int, optional
        目立たせたい path を node id のリストとして与える．

    open_list : list of dict, optional
        探索過程のオープンリスト．リストの要素はノード情報を示す辞書型データ．

    closed_list : list of dict, optional
        探索過程のクローズドリスト．リストの要素はノード情報を示す辞書型データ．

    show_weight : boolean, optional (default: False)
        行列の重みを表示するなら True

    ax : matplotlib.Axes.axes
        描画先．Noneの場合は新規作成する．

    figsize : (int, int), optional (default: (6, 4))
        描画する版面の大きさを指定．suplotsの引数と同じ．

    nscale : float, opeional (default: 1)
        描画するノードの大きさに掛ける係数．例えば，0.8にすれば，0.8倍の大きさになる．

    need_return: bool
        戻り値を得るなら True. 描画のみが目的でその後の処理が不要なら False が無難．

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    G : networkx.classes.digraph.DiGraph
    graph_pos : dict

    Note
    ----
    - need_return = True の場合に限り，戻り値が得られる

    """
    if isinstance(graph_A, netx.classes.digraph.DiGraph):
        G = graph_A
    elif isinstance(graph_A, list) and isinstance(graph_A[0], list) and isinstance(graph_A[0][0], tuple):
        G = netx.DiGraph()
        G.add_weighted_edges_from([
            item for (n, e) in enumerate(graph_A)
                 for item in list(map(lambda x: (n, x[0], x[1]), e))
        ])
    elif isinstance(graph_A, list) and isinstance(graph_A[0], tuple):
        G = netx.DiGraph()
        G.add_weighted_edges_from(graph_A)
    elif isinstance(graph_A, dict):
        G = netx.DiGraph()
        G.add_weighted_edges_from([(k, vv[0], vv[1]) for k, v in graph_A.items() for vv in v])
    elif isinstance(graph_A, list) and isinstance(graph_A[0], list):
        graph_A = np.array(graph_A)
        G = netx.from_numpy_array(graph_A, create_using=netx.DiGraph)
            # - NumPy配列による Adjacency Matrix から，NetworkXグラフ構造体を生成する．
            #   NetworkXグラフ構造体の中身は，気にしなくてよい．
            # - create_usingは，本講義では常に next.DiGraph (有向グラフの意味) を指定すればよい．
            #   - これも，講義の本質とは離れているため，あまり気にしなくてよい．
    else:
        G = netx.to_networkx_graph(graph_A, create_using=netx.DiGraph)

    if ax is not None:
        fig = None
    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
            # - size 1 x 1 で作成すると，axはスカラになります（ax[0]やax[0,0]などの表記がerrorになる）
            #   size N x 1 や N x 1 で作成すると，axは1次元の配列の形になります．（ax や ax[0,0]はerrorになる）
            #   size N x M で作成すると，axは2次元の行列の形になります．（ax[0,0]が使える．ax や ax[0]はerror）
            # - figsizeは，必要に応じて見やすい大きさに変えましょう．
            # - layout指定により aspect ratio = 1 にしておくと，_posの意味が分かりやすいはず

    # Origin is (left, top)
    ax.yaxis.set_inverted(True)
    
    if graph_pos is None:
        graph_pos = netx.spring_layout(G)

    # Draw base graph
    netx.draw(G, graph_pos, ax=ax, connectionstyle='arc3, rad=0.25',
              with_labels=True, node_size=1000*nscale,             # nodeに関する描画設定(1)
              node_color="aqua", edgecolors="blue", linewidths=1,  # nodeに関する描画設定(2)
              width=2, arrowsize=20                                # edgeに関する描画設定
             )
        # - グラフ構造を描く
        #   cf. https://networkx.org/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw.html
        #       https://networkx.org/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html
        # - graph_pos 相当の変数は，自前で設定せずとも，以下の自動レイアウトでもよい．
        #     graph_pos = netx.spring_layout(graph_A)
        #   - ただし，見やすいレイアウトとは限らない．
        #   - その他レイアウトに使える関数は以下を参考に．（使う必要はないし，ここに時間をかけるべきではない．）
        #     https://networkx.org/documentation/stable/reference/drawing.html#module-networkx.drawing.layout

    # Highlight start node(s)
    netx.draw_networkx_nodes(G, graph_pos, ax=ax, nodelist=[0],
                             node_size=1200*nscale, node_shape='8',
                             node_color="aqua", edgecolors="blue", linewidths=2)
        # - draw_netowrkx_nodes(..., nodelist=..., ...) によりことで，指定のノードのみ描画できる
        #   cf. https://networkx.org/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw_networkx_nodes.html
        # - Initial(start) node (常に0番ノード) に，枠をつけたり，別形をつけることで，目立たせる．

    # Highlight goal(target) node(s)
    if graph_target is not None:
        if not isinstance(graph_target, list):
            graph_target = [graph_target]
        netx.draw_networkx_nodes(G, graph_pos, ax=ax, nodelist=graph_target,
                                 node_size=1200*nscale, node_shape='s',
                                 node_color="orange", edgecolors="red", linewidths=2)
            # - Target node も同様に目立たせる

    # Add Edge label for the cost (weight)
    if show_weight:
        edge_labels = netx.get_edge_attributes(G, "weight")
        _custom_draw_networkx_edge_labels(G, graph_pos, ax=ax, edge_labels=edge_labels, label_pos=0.3, rad=0.25)

    # Highlight nodes in open_list
    if open_list is not None and open_list:
        if isinstance(open_list[0], dict):
            _list = [a['id'] for a in open_list]
        else:
            try:
                _list = [a.id for a in open_list]
            except:
                raise ValueError('item in open_list should be access a.id or a["id"]')
            
        netx.draw_networkx_nodes(G, graph_pos, ax=ax, nodelist=_list,
                                 node_size=1200*nscale, node_shape='o',
                                 node_color="aqua", edgecolors="green", linewidths=4)

    # Highlight nodes in closed_list
    if closed_list is not None and closed_list:
        if isinstance(closed_list, dict):
            _list = closed_list.keys()
        elif isinstance(closed_list[0], dict):
            _list = [a['id'] for a in closed_list]
        else:
            try:
                _list = [a.id for a in closed_list]
            except:
                raise ValueError('item in closed_list should be access a.id or a["id"]')
        netx.draw_networkx_nodes(G, graph_pos, ax=ax, nodelist=_list,
                                 node_size=1200*nscale, node_shape='o',
                                 node_color="aqua", edgecolors="red", linewidths=4)

    # Highlight path
    if highlight_path is not None:
        _edgelist = []
        for i, n in enumerate(highlight_path):
            if i == 0: continue
            _edgelist.append((highlight_path[i-1], highlight_path[i]))
        netx.draw_networkx(G, graph_pos, ax=ax, alpha=0.6,
                           nodelist=highlight_path, node_color="orange", linewidths=0, arrowstyle='-', node_size=1800*nscale,
                           edgelist=_edgelist, edge_color="orange", width=6, connectionstyle='arc3, rad=0.25', 
                           )

    # 注意）安易に，ここで，plt.savefig()を利用した画像保存を追加しないようにしてください．
    # ループ内での plt.savefig() 呼び出しは，最悪の場合，ほかの学生の受講の妨げになります．
    # （あなたの端末だけでなく，演習室の全端末がハングアップする可能性があります．）

    if need_return:
        return fig, ax, G, graph_pos


################################################################################
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

class MazeGraph:
    """ 迷路をグラフとして扱うためのクラス

    Attributes
    ----------
    graph_edgedict: dict of list
        グラフの隣接リスト表現

    graph_edgelist: dict of list
        [DEPRECATED] グラフの隣接リスト表現 (graph_edgedictと同じ)

    graph_pos: list (list of tuple)
        グラフの描画位置

    graph_target: int
        グラフ上のゴールに相当するノードID
    
    maze_def : list of list
        迷路の行列表現

    """

    def __init__(self,
                 maze_def=None,
                 maze_goal=None):
        """
        Parameters
        ----------
        maze_def : str
            迷路を定義するための文字列（改行を必要とするためdocstring形式で入力する）
            左上が (0, 0) であり，常にスタート地点である．
        maze_goal: (int, int)
            迷路のゴールを指定する
        """
        
        self.maze_def = None
        self._maze_size = (0, 0)
        self._maze_goal = (-1, -1)
        self.graph_edgedict = {}
        self.graph_edgelist = {}  # deprecated
        self.graph_pos = []
        self.graph_target = None

        if maze_def is not None:
            if isinstance(maze_def, str):
                self.load_from_string(maze_def)
            elif isinstance(maze_def, list):
                self.load_from_matrix(maze_def)
            else:
                raise ValueError('maze_def should be str or matrix(list of lists)')

            self.set_goal(maze_goal)
    
    def set_goal(self, maze_goal):
        """ ゴールを座標で指定する

        Parameters
        ----------
        maze_goal: (int, int)
            迷路のゴールを座標で指定する
        """
        if maze_goal is not None:
            self._maze_goal = maze_goal
            self.graph_target = maze_goal[1] * self.W() + maze_goal[0]
        else:
            self._maze_goal = (-1, -1)
            self.graph_target = None

    def is_goal(self, x, y=None):
        """ ゴール判定

        Parameters
        ----------
        x: int | (int, int)
            x 座標．ただし，tuple の場合は, (x座標, y座標）
        y: int | None
            y 座標．x が tuple で指定されている場合は無視する．

        Returns
        -------
        boolean
            当該座標がゴールならば True
        """
        if isinstance(x, tuple) or isinstance(x, list):
            x, y = x[0], x[1]

        return self._maze_goal == (x, y)

    def pos2nodeid(self, x, y=None):
        """ 迷路の x, y 座標を，グラフのノードIDに変換する

        Parameters
        ----------
        x: int | (int, int)
            x 座標．ただし，tuple の場合は, (x座標, y座標）
        y: int | None
            y 座標．x が tuple で指定されている場合は無視する．

        Returns
        -------
        int
            グラフのノードID
        """
        if isinstance(x, tuple):
            x, y = x[0], x[1]

        return y * self.H + x
    
    def load_from_string(self, maze_def_str: str):
        """迷路を迷路文字列から読み込む
        
        Example
        -------
            >>> maze_def_str = \"\"\"
            1 0 1 1 1 1 1 0 1
            1 0 0 0 0 0 1 0 1
            1 1 1 1 1 1 1 0 1
            1 0 0 0 0 0 1 0 1
            2 0 2 1 2 0 1 1 1
            3 0 0 0 3 0 0 0 1
            4 5 6 7 1 1 1 1 1
            0 0 1 0 1 0 1 0 0
            1 1 1 0 1 0 1 1 1
            \"\"\"
        """
        
        y = -1
        _maze_def = []
        for line in maze_def_str.split('\n'):
            if not line: continue
            y+=1
        
            line_nums = list(map(int, line.split()))
            _maze_def.append(list(line_nums))

        self.load_from_matrix(_maze_def)

    def load_from_matrix(self, maze_def: list):
        """迷路を迷路表現した行列から読み込む

        行列の形式は迷路文字列について，改行と空白で区切って得られる `list of list`
        """
        W, H = len(maze_def[0]), len(maze_def)
        
        graph_edgedict = {}
        graph_pos = {}
        for y, col in enumerate(maze_def):
            for x, v in enumerate(col):
                if v == 0: continue
        
                _list = []
                _id = y * W + x
        
                # Left
                if x > 0 and maze_def[y][x-1] > 0:
                    _list.append((_id-1, maze_def[y][x-1]))
                # Right
                if x < W-1 and maze_def[y][x+1] > 0:
                    _list.append((_id+1, maze_def[y][x+1]))
                # Up
                if y > 0 and maze_def[y-1][x] > 0:
                    _list.append((_id-W, maze_def[y-1][x]))
                # Down
                if y < H-1 and maze_def[y+1][x] > 0:
                    _list.append((_id+W, maze_def[y+1][x]))
        
                if _list:
                    graph_edgedict[_id] = _list
                    graph_pos[_id] = (x, y)
        del x, y

        self.maze_def = maze_def
        self._maze_size = (H, W)

        self.graph_edgedict = graph_edgedict
        self.graph_edgelist = graph_edgedict
        self.graph_pos = graph_pos

    def draw_graph(self, **kwargs):
        """グラフとして描画する
        Parameters
        ----------
        **kwargs
            任意引数は draw_graph_structure() に渡される
        """
        return draw_graph_structure(
            self.graph_edgedict, self.graph_pos, self.graph_target, **kwargs)

    def draw_maze(self, ax=None):
        """迷路として描画する
        Parameters
        ----------
        ax : matplotlib.Axes.axes
            描画先．Noneの場合は新規作成する．

        Note
        ----
        - 重みの色分けをしているが 8 段階までしか対応していない．
        """
        need_return = False
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            need_return = True
        ax.xaxis.set_ticks_position('top')
        ax.yaxis.set_inverted(True)

        mycolor = matplotlib.colormaps['cool'].resampled(8)
        _cmap = ListedColormap(np.vstack(([0., 0., 0., 1.], mycolor(range(8)))))

        # color map
        ax.imshow(self.maze_def, vmin=0, cmap=_cmap, interpolation='none')

        # cell text
        for (j, i), label in np.ndenumerate(self.maze_def):
            ax.text(i, j, label, ha='center', va='center')

        # highlight goal
        ax.scatter(*self._maze_goal, s=4*72, marker='*', facecolor='yellow')

        if need_return:
            return fig, ax
    
    def size(self):
        """迷路のサイズを返す
        
        Returns
        -------
        (int, int)
            Width, Height
        """
        return self._maze_size

    def W(self):
        """迷路の幅（x方向のサイズ）を返す

        Returns
        -------
        int
            Width
        """
        return self._maze_size[0]

    def H(self):
        """迷路の高さ（y方向のサイズ）を返す

        Returns
        -------
        int
            Height
        """
        return self._maze_size[1]

class Robot:
    """ 迷路の上でロボットの絵を動かすためのクラス
    """
    _dir2op = {
        'L': ((-1, 0), 'LEFT'),  'a': ((-1, 0), 'LEFT'),
        'R': ((+1, 0), 'RIGHT'), 'd': ((+1, 0), 'RIGHT'),
        'U': ((0, -1), 'UP'),    'w': ((0, -1), 'UP'),
        'D': ((0, +1), 'DOWN'),  's': ((0, +1), 'DOWN')
    }

    def __init__(self, image_filename, maze, fig=None, ax=None, zoom=1.0, output=None):
        self._position = [0, 0]
        self._accumulated_cost = 0
        self._maze = maze
        self._fig = fig if fig is not None else plt.gcf()
        self._ax = ax if ax is not None else plt.gca()
        self._im = OffsetImage(plt.imread(image_filename), zoom=zoom)

        self._ab = AnnotationBbox(self._im, self._position, xycoords='data', frameon=False)
            # self._position is mutable (pointer) object.
        self._artist = self._ax.add_artist(self._ab)
        self._fig.canvas.draw()

        # assign logging function
        self._output = output
        if isinstance(output, ipywidgets.Label):
            self._output_log = self.__output_log_ipywidgets_label
        elif isinstance(self._output, matplotlib.widgets.TextBox):
            self._output_log = self.__output_log_matplotlib_textbox
        else:
            self._output_log = self.__output_log

        # Start log
        self._output_log('START [0, 0] / 0')

    def __output_log_ipywidgets_label(self, message, append=False):
        if append:
            self._output.value += message
        else:
            self._output.value = 'Robot: ' + message
        print(message)

    def __output_log_matplotlib_textbox(self, message, append=False):
        if append:
            self._output.set_val(self._output.text + message)
        else:
            self._output.set_val(message)

    def __output_log_stdout(self, message, append=False):
        print(message)


    def update_plot(self, no_wait=True):
        """ 画面の更新

        Robot作成時に与えた fig, ax において，ロボットのみを再描画する．

        Note
        ----
        - `req_move` とともに呼び出されるため，通常であれば気にする必要はない．
        """
        # _pos = (self._position[0], self._position[1])
        # self._ab.xybox = _pos

        if self._maze.is_goal(self._position):
            self._output_log(f'\n--- I MADE IT!! / {self._accumulated_cost}', append=True)
            no_wait = False

        if no_wait:
            self._fig.canvas.blit(self._ax.bbox)
        else:
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()

    def _add_cost(self):
        self._accumulated_cost += self._maze.maze_def[self._position[1]][self._position[0]]

    def update_position(self, offset_x:int=0, offset_y:int=0):
        """
        Parameters
        ----------
        offset_x : int
            x方向の移動量
        offset_y : int
            y方向の移動量

        Note
        ----
        - 外部から呼び出すことは期待されていない．`req_XXX` を介して移動すること．
        - 移動に失敗した場合もその場所のコストを負う
        """
        assert np.abs(offset_x) <= 1
        assert np.abs(offset_y) <= 1

        _x = self._position[0] + offset_x
        _y = self._position[1] + offset_y
        xmax, ymax = self._maze.size()
        if (_x < 0) or (_y < 0) or (_x >= xmax) or (_y >= ymax):
            self._add_cost()
            return False

        if self._maze.maze_def[_y][_x] != 0:
            self._position[0] = _x
            self._position[1] = _y
            self._add_cost()
            return True

        self._add_cost()
        return False

    def req_reset(self, event=None):
        self._position[0] = 0
        self._position[1] = 0  # they are mutable object. Don't replace them by pos = [0, 0]!!
        self._accumulated_cost = 0
        self._output_log(f'RESET: {self._position} / {self._accumulated_cost}')

        self.update_plot(no_wait=False)

    def req_move(self, dir: str):
        """ロボットを移動させる
        
        Parameters
        ----------
        dir : str
            移動方向を1文字の大文字か小文字で指定する．
            `L`, `R`, `U`, `D`. あるいは，`a`, `d`, `w`, `s`.
        """
        _op = self._dir2op[dir]
        if self.update_position(*_op[0]):
            self._output_log(f'{_op[1]}: {self._position} / {self._accumulated_cost}')
        else:
            self._output_log(f'{_op[1]}: {self._position} / {self._accumulated_cost} (BUMP!)')
        self.update_plot(no_wait=False)
    
    def req_move_up(self, event=None):
        """ req_move('U') へのエイリアス """
        self.req_move('U')

    def req_move_down(self, event=None):
        """ req_move('D') へのエイリアス """
        self.req_move('D')

    def req_move_left(self, event=None):
        """ req_move('L') へのエイリアス """
        self.req_move('L')

    def req_move_right(self, event=None):
        """ req_move('R') へのエイリアス """
        self.req_move('R')

if __name__ == "__main__":
    ebai_info()
