import numpy as np


def print_header(outID=None, print_time=False):
    if outID is not None:
        filename = '{}.txt'.format(outID)
    else:
        filename = 'log.txt'
    column_titles = ' {Iter:^5s} {f:^8s} {x:^8s} {F:^7s}    | {alpha_min:^6s} {Kappa_1:^6s} {chicg:^8s} {chipg:^8s} {g_cg:^5s} {nI_cg:^5s} {g_pg:^5s} {nI_pg:^5s} {struct}| {itype:>2s}{nS:>5s}  {gradF:^8s} {flag:>4s} {its:>4s} {residual:^8s}  {target:^8s} {d:^7s} | {ctype:^4s} #newZB  {dirder:^8s}  {bak:^4s}  {s:^8s}|'.format(
        Iter='Iter', f='f', x='|x|', F='F', alpha_min='  alpha ', Kappa_1=' Kappa_1 ', chicg='chi_cg', chipg='chi_pg', g_cg='#B_cg',
        nI_cg='|I_cg|', g_pg='#B_pg', nI_pg='|I_pg|', struct=' n-n  n-z  z-n  z-z ',
        itype='type', nS='nVar', gradF='|gradF|', flag='flag', its='its', residual=' Res ', target='tarRes', d='|d|',
        ctype='type', dirder='dirder', bak='bak', s='stepsize')
    if print_time:
        column_titles += '    prox       Newton      LS          PG         CG   |\n'
    else:
        column_titles += '\n'
    with open(filename, "a") as logfile:
        logfile.write(column_titles)


def print_iteration(iteration, fval, normX, F, alpha, Kappa_1, chi_cg,
                    chi_pg, gI_cg, nI_cg, gI_pg, nI_pg, nn, nz, zn, zz, outID=None):
    if outID is not None:
        filename = '{}.txt'.format(outID)
    else:
        filename = 'log.txt'

    if type(alpha) == np.float64:
        alpha_min = alpha_max = alpha
    else:
        alpha_max = max(alpha)
        alpha_min = min(alpha)
    contents = "{it:5d} {fval:8.2e} {normX:8.2e} {F:8.5e} | {alpha_min:8.2e} {Kappa_1:8.2e} {chi_cg:8.2e} {chi_pg:8.2e} {I1:>5d} {nI1:>5d}  {I2:>5d} {nI2:>5d} {nn:>5d}{nz:>5d}{zn:>5d}{zz:>5d} |".format(it=iteration, fval=fval, normX=normX, F=F,
                                                                                                                                                                                                          alpha_min=alpha_min,
                                                                                                                                                                                                          Kappa_1=Kappa_1,
                                                                                                                                                                                                          chi_cg=chi_cg, chi_pg=chi_pg,
                                                                                                                                                                                                          I1=gI_cg, nI1=nI_cg,
                                                                                                                                                                                                          I2=gI_pg, nI2=nI_pg,
                                                                                                                                                                                                          nn=nn, nz=nz, zn=zn, zz=zz)
    with open(filename, "a") as logfile:
        logfile.write(contents)
