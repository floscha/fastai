"Cleaning and feature engineering functions for structured data"
from ..torch_core import *

__all__ = ['Categorify', 'FillMissing', 'FillStrategy', 'Normalize', 'TabularProc', 'RemoveMinVariance']

@dataclass
class TabularProc():
    "A transform for tabular dataframes."
    cat_names:StrList
    cont_names:StrList

    def __call__(self, df:DataFrame, test:bool=False):
        "Apply the correct function to `df` depending on `test`."
        func = self.apply_test if test else self.apply_train
        func(df)

    def apply_train(self, df:DataFrame):
        "Function applied to `df` if it's the train set."
        raise NotImplementedError
    def apply_test(self, df:DataFrame):
        "Function applied to `df` if it's the test set."
        self.apply_train(df)

class Categorify(TabularProc):
    "Transform the categorical variables to that type."

    def apply_train(self, df:DataFrame):
        self.categories = {}
        for n in self.cat_names:
            df.loc[:,n] = df.loc[:,n].astype('category').cat.as_ordered()
            self.categories[n] = df[n].cat.categories

    def apply_test(self, df:DataFrame):
        for n in self.cat_names:
            df.loc[:,n] = pd.Categorical(df[n], categories=self.categories[n], ordered=True)

FillStrategy = IntEnum('FillStrategy', 'MEDIAN COMMON CONSTANT')

@dataclass
class FillMissing(TabularProc):
    "Fill the missing values in continuous columns."
    fill_strategy:FillStrategy=FillStrategy.MEDIAN
    add_col:bool=True
    fill_val:float=0.

    def apply_train(self, df:DataFrame):
        self.na_dict = {}
        for name in self.cont_names:
            if pd.isnull(df.loc[:,name]).sum():
                if self.add_col:
                    df.loc[:,name+'_na'] = pd.isnull(df.loc[:,name])
                    if name+'_na' not in self.cat_names: self.cat_names.append(name+'_na')
                if self.fill_strategy == FillStrategy.MEDIAN: filler = df.loc[:,name].median()
                elif self.fill_strategy == FillStrategy.CONSTANT: filler = self.fill_val
                else: filler = df.loc[:,name].dropna().value_counts().idxmax()
                df.loc[:,name] = df.loc[:,name].fillna(filler)
                self.na_dict[name] = filler

    def apply_test(self, df:DataFrame):
        for name in self.cont_names:
            if name in self.na_dict:
                if self.add_col:
                    df.loc[:,name+'_na'] = pd.isnull(df[name])
                    if name+'_na' not in self.cat_names: self.cat_names.append(name+'_na')
                df.loc[:,name] = df.loc[:,name].fillna(self.na_dict[name])

class Normalize(TabularProc):
    "Transform the categorical variables to that type."

    def apply_train(self, df:DataFrame):
        self.means,self.stds = {},{}
        for n in self.cont_names:
            self.means[n],self.stds[n] = df.loc[:,n].mean(),df.loc[:,n].std()
            df.loc[:,n] = (df.loc[:,n]-self.means[n]) / (1e-7 + self.stds[n])

    def apply_test(self, df:DataFrame):
        for n in self.cont_names:
            df.loc[:,n] = (df.loc[:,n]-self.means[n]) / (1e-7 + self.stds[n])

def get_freq_ratio(column:Series):
    """Compute frequency ratio."""
    value_counts = column.value_counts(normalize=True)
    if len(value_counts) == 1:
        return float('inf')
    most_common_freq, second_most_common_freq = value_counts[:2]
    frequency_ratio = most_common_freq - second_most_common_freq
    return frequency_ratio

def get_percent_of_uniq_vals(column:Series):
    """Compute percent of unique values."""
    return len(column.unique()) / len(column) * 100

@dataclass
class RemoveMinVariance(TabularProc):
    """Remove variables below a certain variance threshold.

    For continuous:
    Simple variance
    <link to TransmogrifAI>

    For categorical:
    Near-zero-variance
    http://topepo.github.io/caret/pre-processing.html#nzv

    The default parameters for `freq_cut` and `uniq_cut` are taken from:
    https://github.com/topepo/caret/blob/6546939345fe10649cefcbfee55d58fb682bc902/pkg/caret/R/nearZeroVar.R#L90
    """
    min_var:float=0.00001
    freq_cut:float=95/5
    uniq_cut:float=10.0
    remove_cols:bool=True
    check_sample:float=1.0
    verbose:bool=True

    def apply_train(self, df:DataFrame):
        self.cols_to_drop = []
        for n in self.cont_names:
            col = df[n]
            if self.check_sample < 1.0: col = col.sample(int(len(col) * check_sample))
            n_variance = col.var()
            if n_variance < self.min_var:
                if self.verbose:
                    if self.remove_cols: print(("Dropping column '%s' since its variance of %0.4f"
                                                + " is below the threshold") % (n, n_variance))
                    else:                print(("Attention: The variance of column '%s' is %0.4f"
                                                + " and thus below the threshold") % (n, n_variance))
                if self.remove_cols: self.cols_to_drop.append(n)
        for n in self.cat_names:
            col = df[n]
            if get_freq_ratio(col) > self.freq_cut and get_percent_of_uniq_vals(col) < self.uniq_cut:
                if self.verbose: print("Near-zero-variance")
                if self.remove_cols: self.cols_to_drop.append(n)
        df.drop(columns=self.cols_to_drop, inplace=True)

    def apply_test(self, df:DataFrame):
        df.drop(columns=self.cols_to_drop, inplace=True)
