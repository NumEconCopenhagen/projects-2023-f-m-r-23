from dstapi import DstApi

def import_REG():

    # connecting to dst
    ind = DstApi('REGK11')

    # choosing which variables to import
    params = {'table':'REGK11',
         'format':"BULK",
         'variables':[{'code':'OMRÅDE','values':['*']},
                     {'code':'PRISENHED','values':['INDL']},
                     {'code':'DRANST','values':['1']},
                     {'code':'FUNK1','values':['4']},
                     {'code':'ART','values':['TOT']},
                     {'code':'TID','values':['>2007']}]}
    # importing data
    data = ind.get_data(params)

    # dropping irrelevant and/or constant columns
    data.drop(columns=['PRISENHED', 'DRANST', 'FUNK1', 'ART'],inplace=True)

    # renaming columns
    data.rename(columns={'INDHOLD':'SUNDHEDSUDGIFTER'},inplace=True)

    return data