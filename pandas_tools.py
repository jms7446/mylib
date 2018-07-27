#!/usr/bin/env python

import sys

import pandas as pd
from IPython.display import *
import matplotlib.pyplot as plt
import numpy as np
import json

import common_tools as cmt

pd.set_option('display.width', 2000)
pd.set_option('max_colwidth', 0)
pd.set_option('max_rows', 200)


###############################################################################
# DataFrame
###############################################################################

def df_to_bdb(df, key_field, value_field, path, mode="w"):
    it = ((cmt.unicode_to_str(row[key_field]), cmt.unicode_to_str(row[value_field])) for idx, row in df.iterrows())
    cmt.iter_to_bdb(it, path, mode=mode)


def split_key_score(x, sep):
    k, v = x.split(sep)
    v = float(v)
    return [k, v]


def report_stat(names, values, transpose=True):
    if transpose:
        values = zip(*values)
    df = pd.DataFrame(values, columns=names)
    return df
    
    
def report_stat_one_line(names, value):
    values = [value]
    return report_stat(names, values, transpose=False)
    

def report_value_counts(sr, bins=None, sort_index=False, show_ratio=False):
    name = sr.name
    df = pd.DataFrame(sr.value_counts(bins=bins)).reset_index()
    df = df.rename(columns={name: "count"})
    df = df.rename(columns={"index": name})    
    if sort_index:
        df = df.sort_values(name).reset_index(drop=True)
    if show_ratio:
        df["ratio"] = 100.0 * df["count"] / df["count"].sum()
    return df
    
    
def plot_function(f, xs, ax=None):
    if ax is None:
        ax = plt.axes()
    A = pd.DataFrame(xs, columns=["x"])
    A["y"] = A["x"].map(f)
    A.set_index("x")["y"].plot(ax=ax)
    

def move_column(A, col, idx):
    """column 의 위치를 바꾼 DataFrame 을 반환한다.

    Parameters
    ----------
    A : DataFrame
    col : str, name of column which will be moved
    idx : int,

    Returns
    ------
    B : DataFrame
    """
    columns = A.columns.tolist()
    columns.pop(columns.index(col))
    columns.insert(idx, col)
    return A[columns]


def _remake_columns(old_columns, col_str):
    """Return reordered columns"""
    raw_columns = map(lambda x: x.strip(), col_str.split(","))
    drop_columns = map(lambda x: x[1:], filter(lambda x: x.startswith("-"), raw_columns))
    new_columns = filter(lambda x: not x.startswith("-") and x not in drop_columns, raw_columns)
    all_column_set = set(new_columns + drop_columns)
    rest_columns = filter(lambda x: x not in all_column_set, old_columns)
    new_columns = cmt.flatten_list(map(lambda x: rest_columns if x == "*" else [x], new_columns))
    return new_columns


def reorder_columns(A, col_str):
    """column 을 재정렬한 DataFrame 을 반환한다.

    slicing 을 이용하고, 이는 복제가 일어나므로 큰 DataFrame 에는 자주 사용하지 않는 것이 좋다.

    Parameters
    ----------
    A : DataFrame
    col_str : str,
        use ',' as delimiter
        ex) "name, age, *, -height, -age"
            '*'는 col_str 에 표현되지 않은 나머지 column 을 의미한다.
            '*'를 사용하지 않은 경우, col_str 에 표현되지 않은 column 은 제거된다.
            '-'를 앞에 붙이면 해당 column 은 제거한다.
            age와 같이 '+', '-' 두 곳에 모두 존재하면 '-'로 간주하여 제거한다.


    Returns
    -------
    reordered : DataFrame
    """
    old_columns = A.columns.tolist()
    new_columns = _remake_columns(old_columns, col_str)
    
    return A[new_columns]


def cross_map(xs, ys, f, x_name="x", y_name="y"):
    """xs의 x와 ys의 y에 f를 적용한 결과로 DataFrame을 만들어 반환한다.

    Parameters
    ----------
    xs : list-like, of x
    ys : list-like, of y
    f : (x, y) => z
    x_name : str, will be index name
    y_name : str, will be columns name

    Returns
    -------
    df : DataFrame,
        * index : xs,
        * columns : ys,
        * value : f(x, y)
    """

    l = []
    for x in xs:
        for y in ys:
            l.append((x, y, f(x, y)))
    A = pd.DataFrame(l, columns=[x_name, y_name, "z"])
    B = pd.pivot_table(A, index=x_name, columns=y_name)["z"]
    return B


def flatten_column(A, col, split_func, name=None, drop=True, reset_index=True):
    """column fn을 split_func 을 적용하여 나누어 여러개의 row 로 만든 DataFrame 을 돌려준다.

    Parameters
    ----------
    A : DataFrame
    col : str, source column name
    split_func : object => list-like,
    name : str, default None
        * None 이면 col 을 덮어쓴다.
        * col 과 같으면 덮어쓴다.
    drop : boolean, whether drop col or not

    Returns
    -------
    df : DataFrame
    """
    if name is None:
        name = col
    Y = pd.DataFrame([[i, x]
                      for i, y in A[col].apply(split_func).iteritems()
                      for x in y], columns=['I', name])
    Y = Y.set_index('I')
    if drop or name == col:
        A = A.drop(col, axis=1)
    if reset_index:
        A = A.reset_index(drop=True)
    return A.join(Y)


def split_column(A, col, split_func, names, fill=None, drop=True, reset_index=True):
    """col의 값에 split_func을 사용해 나눈 값들로 새로운 column들이 추가된 DataFrame을 반환한다.

    Parameters
    ----------
    A : DataFrame
    col : str, column name of DataFrame A (src)
    names : list-like, names of splited columns (dest)
    split_func : object => list-like, split function
    fill : object, for padding
    drop : boolean, whether drop col or not

    Return
    ------
    splited : DataFrame
    """
    if reset_index:
        A = A.reset_index(drop=True)
    split_func_with_padding = lambda x: cmt.add_padding_list(split_func(x), len(names), fill)
    B = pd.DataFrame(A[col].map(split_func_with_padding).tolist(), columns=names)
    if drop:
        A = A.drop(col, axis=1)
    return A.join(B)


def split_columns_from_dic(A, col, drop=True):
    B = A.join(pd.DataFrame.from_records(A[col].map(json.loads)))
    if drop:
        B = B.drop(col, axis=1)
    return B


def page(df, p=0, n=10):
    """DataFrame을 page로 나눈 DataFrame 부분을 반환한다.

    Parameters
    ----------
    p : int, page number, 0 is the first page
    n : int, number of rows in a page

    Returns
    -------
    paged : DataFrame
    """

    if len(df) > 0:
        s = n*p
        e = min(n*(p+1), len(df))
        return df.iloc[s:e]
    else:
        return df


def shuffle(df, reset_index=False, seed=None):
    """row 를 무작위로 섞은 DataFrame을 반환한다.

    Parameters
    ----------
    df : DataFrame
    reset_index : boolean,
    seed : int, random seed

    Returns
    -------
    df : DataFrame
    """
    if reset_index:
        df = df.reset_index(drop=True)
    elif np.any(df.index.duplicated()):
        sys.stderr.write("index is duplicated, remove duplicated index or set parameter reset_index=True\n") 
        sys.stderr.write("  => pdt.shuffle(X, reset_index=True)\n")
        raise ValueError("duplicated index")
    np.random.seed(seed)
    return df.reindex(np.random.permutation(df.index))


def divide(X, r, seed=None):
    """X를 랜덤하게 두개의 DataFrame 으로 나누어 DataFrame tuple을 반환한다.

    Parameters
    ----------
    X : DataFrame
    r : double, split ratio
        * r : (1-r) 비율로 데이터를 나눈다.
    seed: random seed

    Returns
    -------
    (df1, df2): tuple of DataFrame
    """
    if seed is not None:
        np.random.seed(seed)
    X = shuffle(X)
    d = int(len(X)*r)

    return X.iloc[:d], X.iloc[d:]


def sample(df, k, replace=False, seed=None):
    """k개를 랜덤하게 뽑아서 돌려준다.

    Parameters
    ----------
    df : DataFrame
    k : int,
    replace : boolean, 복원 또는 비복원 추출
    seed: random seed

    Returns
    -------
    df : DataFrame
    """
    if replace:
        np.random.seed(seed)
        return df.iloc[np.random.randint(0, len(df), size=k)]
    else:
        return shuffle(df, seed=seed).head(k)
    
    
def sample_with_count(A, sample_rate, count_col, sampled_col="sampled", seed=None):
    """각각 다른 횟수만큼 실행해서 성공한 횟수가 1개 이상인 데이터를 돌려준다.

    Parameters
    ----------
    count_col : 시도할 횟수
    sample_rate : 성공 확률
    sampled_col : 성공한 횟수를 기록할 column
    seed : random seed

    Returns
    -------
    df : DataFrame, 성공 횟수가 1번 이상인 데이터    
    """
    np.random.seed(seed)
    s = A[count_col].map(lambda x: np.random.binomial(x, sample_rate))
    R = A[s > 0].copy()
    R[sampled_col] = s
    
    return R


# 제거 예정
def sample_with_count_old(X, count_col, sample_rate, seed=None):
    """sample_rate의 확률로 N번 시도했을 때 한 번 이상 뽑히는 데이터만 남긴다.

    Parameters
    ----------

    Returns
    -------
    """
    counts = X[count_col]
    selected = calc_select_prob_with_count(counts, sample_rate, seed)
    return X[selected]


# 제거 예정
def calc_select_prob_with_count(counts, sample_rate, seed=None):
    """sample_rate의 확률로 N번 시도했을 때 한 번 이상 뽑히는지 여부를 돌려준다.
    중복 item이 있는 데이터에서 item을 sample_rate의 확률로 뽑고, 중복으로 뽑힌 item은 제거한 결과를 빠르게 계산하기 위해
    Parameters
    ----------

    Returns
    -------
    """
    if seed is not None:
        np.random.seed(seed)
    r = np.random.random(len(counts))
    selected = r < (1.0 - (1.0 - sample_rate)**counts)
    return selected


def iter_json(df):
    for idx, row in df.iterrows():
        yield row.to_json() + "\n"


###############################################################################
# display
###############################################################################

# dataframe 표시를 browser 크기만큼 확대할 수 있다.
def expand_total_width(width="100%"):
    from IPython.core.display import display, HTML
    display(HTML("<style>.container { width:%s !important; }</style>" % width))

###############################################################################


###############################################################################
# dataframe display
###############################################################################


def raw_float_frmt(x, big_cnt=2, small_cnt=5, thres=1e3):
    if x > thres:
        fm = '{:,.%sf}' % big_cnt
    else:
        fm = '{:,.%sf}' % small_cnt
    return  fm.format(x)

default_float_frmt = lambda x: raw_float_frmt(x)
default_int_frmt = lambda x: '{:,}'.format(x)


def display_df_for_stat(df, int_frmt=None, float_frmt=None, small_cnt=5, **kwargs):
    if int_frmt is None:
        int_frmt = default_int_frmt
    if float_frmt is None:
        float_frmt = lambda x: raw_float_frmt(x, small_cnt=small_cnt)
        
    frmt_map = {np.dtype('int64'):int_frmt, np.dtype('float64'):float_frmt}
    frmt = {col:frmt_map[df.dtypes[col]] for col in df.columns if df.dtypes[col] in frmt_map.keys()}
    display_df(df, style_class="right_aligned_df", formatters=frmt, **kwargs)

    
def show_style():
    html = HTML('''<style>
                    .right_aligned_df td { text-align: right; }
                    .left_aligned_df td { text-align: right; }
                    .pink_df { background-color: pink; }
                </style>''')
    display(html)

    
def display_or_return_df(df, return_df, style_class=None, formatters=None):
    if return_df:
        return df
    else:
        # style 을 사용하기 위해서는 show_style() 을 실행한 cell이 하나 있어야 함. 
        html = df.to_html(escape=False, classes=style_class, formatters=formatters)
        display(HTML(html))
        return None

        
def display_df(df, col_str="*", img_fields=[], link_fields=[], img_link_fields=[], font_fields=[], n=100,
               img_width=200, img_height=None, return_df=False, style_class=None, formatters=None):
    """DataFrame 을 Html 로 변환하여 화면에 표시

    이미지를 보여주거나, 하이퍼링크를 사용할 수 있고, 아주 넓거나 긴 DataFrame이라도 있는 그대로 화면에 표현한다.

    Parameters
    ----------
    df : DataFrame
    col_str : str, 표시하고자 하는 필드 전체 명세
    img_fields : list-like, 이미지로 보여줄 필드
    link_fields : list-like, 하이퍼링크로 보여줄 필드
    img_link_fields : list-like of tuple, 이미지로 보여주면서, 클릭했을 때 연결되는 링크 페이지를 따로 줄 수 있음.
        * [(img_field, link_field), ...]
        * ex. img_link_fields=[("thumb", "url")]
        * img_field 의 이름으로 표시되며, link_field 에 해당하는 필드는 따로 표시되지 않음
    n : int, 아무리 큰 데이터라도 화면에 다 표시하므로, 실수로 너무 큰 데이터를 받았을 경우 문제가 될 수 있다. 기본값은 안전장치
    img_width, img_height : int, 표시될 이미지의 가로, 세로
    return_df : boolean
        * true : display 하지 않고 DataFrame 을 반환한다.
        * false : display 하고 None 을 반환한다.

    Returns
    -------
    df : if return_df DataFrame else None
    """

    df = df.head(n).copy()

    if len(df) == 0:
        print("Empty Dataframe")
        return

    # 여러개 이미지 필드가 있더라도, 크기는 하나로 통일한다.
    width_str = ' width="%s"' % img_width if img_width is not None else ''
    height_str = ' height="%s"' % img_height if img_height is not None else ''

    def to_img_src(img_url, link, width=None, height=None, target_name=''):
        img_src = "<a href='%s' target=img_%s><img src='%s'%s%s></a>" % (img_url, target_name, img_url, width_str, height_str)
        return img_src

    def to_link_src(link_url, target_name=''):
        link_src = '<a href="%s" target="link_%s">link</a>' % (link_url, target_name)
        return link_src

    fields = _remake_columns(df, col_str)
    
    # 아래는 필드별 가공이 필요한 것들에 대한 처리
    for field in img_fields:
        df[field] = df[field].map(lambda x: "<a href='%s' target=img_%s><img src='%s'%s%s></a>" % (x, field, x, width_str, height_str))

    for field in link_fields:
        df[field] = df[field].map(lambda x: '<a href="%s" target="link_%s">link</a>' % (x, field))

    for (img_field, link_field) in img_link_fields:
        df[img_field] = df.apply(lambda x: "<a href='%s' target=img_%s><img src='%s'%s%s></a>" % (x[link_field], img_field, x[img_field], width_str, height_str), axis=1)
        if img_field != link_field:
            fields.pop(fields.index(link_field))
        else:
            sys.stderr.write("두 개 필드가 동일할 경우, img_fields=['%s'] 을 사용하시길 추천합니다." % img_field)
            sys.stderr.flush()
        
    for field, d in font_fields:
        size = d.get("size", 10)
        color = d.get("color", "black")
        cond = d.get("cond", None)
        df[field] = df[field].map(lambda x: "<font size=%s color='%s'>%s</font>" % (size, color, x) if (cond is None or cond(x)) else x)

    df = df[fields]
    return display_or_return_df(df, return_df=return_df, style_class=style_class, formatters=formatters)


def display_dfs(dfs, return_df=False, axis=1, formatters=None, style_class=None):
    """여러개의 DataFrame 을 묶어서 화면에 보여준다.

    note. 강제로 concat 하기 위해서 index 를 초기화한다.

    Parameters
    ----------
    dfs : list-like of DataFrame
    return_df : boolean
        * true : display 하지 않고 DataFrame 을 반환한다.
        * false : display 하고 None 을 반환한다.
    axis : {0, 1}
    """
    dfs = map(lambda df: df.reset_index(drop=True), dfs)
    df = pd.concat(dfs, axis=axis)
    return display_or_return_df(df, return_df=return_df, style_class=style_class, formatters=formatters)


# FIXME 여기 함수가 상당히 산만하다. 간결하게 수정되었으면 함
# rank가 int였다가, merge 이후 None 값이 들어가면서 float64로 바뀌고, float의 정수화시에 생기는 오차가 신경쓰인다. 
# sys.maxint 를 쓰지 않는 것도 이것과 연관됨
def calc_rel_rank(x, y, simplify_diff):
    def simplify_rank_move(s):
        rank_max = 10000 # sys.maxint  # 적당히 높은 값
        s = s.fillna(rank_max) #.astype(int)
        l = list(s.values)
        mis = set(cmt.maximum_increasing_subsequence(l)) - set([rank_max])
        def _convert_rank_move(x):
            if x in mis:
                return ""
            elif x == rank_max:
                return "-"
            else:
                return "%s" % int(round(x))
        return s.map(_convert_rank_move)

    # join을 위해서 DataFrame으로 만들어서 사용한다. 
    # 더 간결한 방법이 있으면 수정할 것. 
    id_col = "id"      # 여기서만 사용되고 버려진다. 
    rank_col = "rank"  # 여기서만 사용되고 버려진다. 
    x = pd.DataFrame({id_col: x.copy()})
    y = pd.DataFrame({id_col: y.copy()})
    y[rank_col] = range(len(y))
    # x의 index로 맞춘다. 
    r = pd.merge(x, y, on=id_col, how="left")[rank_col]
    if simplify_diff:
        r = simplify_rank_move(r)
    return r
    
    
def display_df_pair(A, B, on, rel_rank_col="rel_rank", simplify_diff=True, col_str="*", col_str_a=None, col_str_b=None, return_df=False, **kwargs):
    """두 개의 DataFrame 을 묶어서 화면에 보여준다. (동일한 구조의 DataFrame을 가정한다.)

    두 DataFrame 의 차이를 비교하기 위한 목적에 특화됨. 
    axis=1 로 concat 하는 것만 제공한다. 
    note. 강제로 concat 하기 위해서 index 를 초기화한다.

    Parameters
    ----------
    A : DataFrame
    B : DataFrame
    on : str, 두 DataFrame 을 비교하기 위해 join 할 때, key 가 되는 col
    rel_rank_col : str, 상대편의 랭크 (0부터 시작, DataFrame 의 index와 동일한 값으로 두는 것이 더 직관적이어서)
    simplify_diff : boolean, 의미있는 변화만 남기고 나머지는 제거
    kwargs : display_df 로 넘기는 파라미터
    """
    A = A.reset_index(drop=True).copy()
    B = B.reset_index(drop=True).copy()
    
    A[rel_rank_col] = calc_rel_rank(A[on], B[on], simplify_diff)
    B[rel_rank_col] = calc_rel_rank(B[on], A[on], simplify_diff)
    
    if "fond_fields" not in kwargs:
        kwargs["font_fields"] = [("rel_rank", {"size": 6, "color": "red"})]
    if col_str_a is None:
        col_str_a = col_str
    if col_str_b is None:
        col_str_b = col_str
    col_str_a = "%s, %s" % (rel_rank_col, col_str_a)
    col_str_b = "%s, %s" % (rel_rank_col, col_str_b)
    A = display_df(A, col_str=col_str_a, return_df=True, **kwargs)
    B = display_df(B, col_str=col_str_b, return_df=True, **kwargs)
    
    
    return display_dfs([A, B], return_df=return_df)


###############################################################################
# plotting
###############################################################################

def hist_by_quantile(S, a=0.01, **kwargs):
    quantile_upper = S.quantile(1.0 - a)
    quantile_lower = S.quantile(a)
    S[S < quantile_upper][S > quantile_lower].hist(**kwargs)


def subplots_axs(num_axs, col_wrap=4, unit_size=(4, 3), figsize=None, **kwargs):
    # plot count
    num_col = min(num_axs, col_wrap)
    num_row = (num_axs - 1) / num_col + 1

    # plot size
    if figsize is None:
        figsize = (1 + unit_size[0] * num_col, 1 + unit_size[1] * num_row)

    # create axs
    fig, axs = plt.subplots(num_row, num_col, figsize=figsize, **kwargs)
    if num_row > 1 and num_col > 1:
        axs = cmt.flatten_list(axs)
    return axs


from sklearn import metrics
def plot_precision_recall_curves(target, feature, ax, sample_weight=None, color='blue', fn=''):
    pr, rc, thresholds = metrics.precision_recall_curve(target, feature, pos_label=1, sample_weight=sample_weight)
    average_precision_score = metrics.average_precision_score(target, feature, sample_weight=sample_weight)
    ax.plot(rc, pr, label='%s : %.3f' % (fn, average_precision_score), color=color)
    ax.set_xlabel('Recal')
    ax.set_ylabel('Precision')
    ax.legend(loc='best')


def plot_precision_recall_curves_with_thres(target, feature, ax, sample_weight=None, fn=''):
    pr, rc, thresholds = metrics.precision_recall_curve(target, feature, pos_label=1, sample_weight=sample_weight)
    ax.plot(thresholds, pr[:-1], label='precision', color='red')
    ax.plot(thresholds, rc[:-1], label='recall', color='blue')
    ax.set_xlabel('Thresholds')
    ax.set_ylabel('Precision/Recall')
    ax.legend(loc='best')


def plot_roc_curve(target, feature, ax, sample_weight=None, color='blue', fn=''):
    fpr, tpr, thresholds = metrics.roc_curve(target, feature, pos_label=1, sample_weight=sample_weight)
    ax.plot(fpr, tpr, label='%s' % fn, color=color)
    ax.set_xlabel('fpr')
    ax.set_ylabel('tpr')
    ax.legend(loc='best')


def plot_roc_curve_with_thres(target, feature, ax, sample_weight=None, fn=''):
    fpr, tpr, thresholds = metrics.roc_curve(target, feature, pos_label=1, sample_weight=sample_weight)
    ax.plot(thresholds, fpr, label='fpr', color='red')
    ax.plot(thresholds, tpr, label='tpr', color='blue')
    ax.set_xlabel('thresholds')
    ax.set_ylabel('tpr/fpr')
    ax.legend(loc='best')


def get_log_odds(target, feature, bins, f_range=None, M=10):
    """return log ( P(feature=x | target=1) / P(feature=x | target=0) )

       tn : target name, 0 or 1
       fn : x name
       f_range : x의 범위 제한
       M : smoothing factor
    """
    tn = target.name
    fn = feature.name
    X = pd.concat([target, feature], axis=1)
    if f_range is not None:
        X = X[(X[fn]>f_range[0]) & (X[fn]<f_range[1])]
    X['_cut'] = pd.cut(X[fn], bins=bins).astype("string")
    X['_cut'] = X._cut.map(lambda x: float(x.split(',')[0][1:]))
    Y = X.groupby('_cut').apply(lambda x: np.log((x[tn].sum() + 1.0*M/bins) / ((1.0-x[tn]).sum() + 1.0*M/bins)))
    #    display(X.groupby('_cut').apply(lambda x: (x[tn].sum(), (1-x[tn]).sum())))
    #    display(Y)
    Y = Y - np.log( (1.0 * X[tn].sum() + M) / ( (1.0-X[tn]).sum() + M) )
    Y = pd.DataFrame(Y, columns=['%s_log_odds' % fn])
    return Y


def plot_log_odds(target, feature, bins, f_range=None, M=10, figsize=(10, 3), f_min=None, f_max=None):
    alt_feature = feature.copy()
    if f_min is not None:
        alt_feature = feature.map(lambda x: x if x >= f_min else f_min)
    if f_max is not None:
        alt_feature = feature.map(lambda x: x if x <= f_max else f_max)
    LO = get_log_odds(target, alt_feature, bins, f_range=f_range, M=M)
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    ax = axs[0]; alt_feature[target==1].hist(bins=bins, alpha=0.4, color='red', normed=True, range=f_range, ax=ax)
    ax = axs[0]; alt_feature[target==0].hist(bins=bins, alpha=0.4, color='blue', normed=True, range=f_range, ax=ax)
    ax = axs[1]; LO.plot(ax=ax)
