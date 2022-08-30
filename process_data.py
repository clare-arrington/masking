import pandas as pd

# min_count - requires target to show up n times or its cut out
# min length - requires sentence to be above length k, important for context window
# occurence limit - this flag makes it so that one sentence doesn't have too many targets in it

def filter_target_data(
    target_paths,
    targets=None,
    min_count=50,
    min_length=25, 
    occurence_limit=10, 
    subset_num=12500
    ):
    
    all_data = []
    for corpus, path in target_paths.items():
      data = pd.read_pickle(path)
      print(f'= {len(data):,} rows pulled for {corpus} =')

      ## Do some initial filtering before combining
      if targets is not None:
          data = data[data.target.isin(targets)]
          print(f'\n{len(data):,} target instances after targets selected')

      ## Apply minimum length
      len_data = data[data.length >= min_length]
      ids = len_data.sent_idx.unique()

      data = data[data.sent_idx.isin(ids)]
      print(f'{len(data):,} instances after {min_length} length minimum applied')
      del ids

      ## Apply occurence limit
      index_vc = data.sent_idx.value_counts()
      ids = index_vc[index_vc <= occurence_limit].index
      data = data[data.sent_idx.isin(ids)]
      print(f'{len(data):,} after {occurence_limit} occurence limit applied\n\n')

      all_data.append(data)

    data = pd.concat(all_data)
    del all_data
    print(f'{len(data):,} target instances pulled')

    og_vc = len(data.target.unique())
    print(f'\n== {og_vc} targets before anything else removed ==')

    ## Subset data
    vc = data.target.value_counts()
    too_many = vc[vc > subset_num]
    for target in too_many.index:
        # TODO: maybe this doesnt work bc of multiindex
        index_vc = data.sent_idx.value_counts()
        ids = data[data.target == target].sent_idx
        
        # This selects from the bottom up (so smallest to largest)
        bigger_rows = index_vc[index_vc.index.isin(ids)][:-subset_num]
        
        print(f'\t{len(bigger_rows):,} rows being removed for {target}')
        data = data[~data.sent_idx.isin(bigger_rows.index)]
    print(f'{len(data):,} after {subset_num} max applied')

    ## Apply minimum count
    vc = data.target.value_counts()
    targets = vc[vc >= min_count].index
    data = data[data.target.isin(targets)]
    print(f'{len(data):,} after insufficient targets removed')
    
    new_vc = len(vc)
    print(f'\n== {new_vc} targets left after filtering ==')
    print(f'{og_vc - new_vc} were removed')

    return data

