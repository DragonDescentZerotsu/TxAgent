"""Semantic display names for BBB transporter identifiers.

Raw source identifiers remain unchanged.  This frozen registry only supplies a
canonical comparison/display field; database collisions therefore stay visible
instead of being silently reinterpreted from nearby prose.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


TRANSPORTER_IDENTIFIER_VERSION = "bbb_transporter_identifier.v2"
# Snapshots queried from UniProtKB and NCBI Gene on 2026-09-04.
UNIPROT_NAMES = {
    'A0A059J0G5': 'ABC multidrug transporter MDR1 [Trichophyton interdigitale; UniProt A0A059J0G5]',
    'A0A059JJ46': 'ABC multidrug transporter MDR2 [Trichophyton interdigitale; UniProt A0A059JJ46]',
    'A0A059JK44': 'MDR4 - ABC multidrug transporter MDR2 [Trichophyton interdigitale; UniProt A0A059JK44]',
    'A0A084B9Z0': 'Short-chain dehydrogenase/reductase SAT2 [Stachybotrys chartarum; UniProt A0A084B9Z0]',
    'A0A0B5A886': 'GP - Envelopment polyprotein [SFTS phlebovirus; UniProt A0A0B5A886]',
    'A0A0G2K309': 'Slc25a15 - Mitochondrial ornithine transporter 1 [rat; UniProt A0A0G2K309]',
    'A0A1C7E424': 'mdrP - Na(+), Li(+), K(+)/H(+) antiporter [Planococcus maritimus; UniProt A0A1C7E424]',
    'A0A1U8QT10': 'ABC multidrug transporter atrA [Emericella nidulans; UniProt A0A1U8QT10]',
    'A0B7J8': 'Mthe_0884 - Phosphoglycolate phosphatase [Methanothrix thermoacetophila; UniProt A0B7J8]',
    'A0K4E0': 'mtgA - Biosynthetic peptidoglycan transglycosylase [Burkholderia cenocepacia; UniProt A0K4E0]',
    'A1L1P9': 'slc47a1 - Multidrug and toxin extrusion protein 1 [zebrafish; UniProt A1L1P9]',
    'A1L272': 'slc29a4 - Equilibrative nucleoside transporter 4 [zebrafish; UniProt A1L272]',
    'A6QLW8': 'SLC22A7 - Solute carrier family 22 member 7 [bovine; UniProt A6QLW8]',
    'A7KVC2': 'ABC transporter C family MRP4 [Zea mays; UniProt A7KVC2]',
    'A7MBE0': 'SLC22A1 - Solute carrier family 22 member 1 [bovine; UniProt A7MBE0]',
    'A8XWB7': 'vps-28 - Vacuolar protein sorting-associated protein 28 homolog [C. briggsae; UniProt A8XWB7]',
    'A8Y3Q1': 'mrps-5 - Small ribosomal subunit protein uS5m [C. briggsae; UniProt A8Y3Q1]',
    'A9CB25': 'SLC22A4 - Solute carrier family 22 member 4 [olive baboon; UniProt A9CB25]',
    'A9ULC7': 'slc51a - Organic solute transporter subunit alpha [western clawed frog; UniProt A9ULC7]',
    'B0S6T2': 'slc15a2 - Solute carrier family 15 member 2 [zebrafish; UniProt B0S6T2]',
    'B2KWH4': 'ABC1 - ABC transporter 1 [Ajellomyces capsulatus; UniProt B2KWH4]',
    'B2RX12': 'Abcc3 - ATP-binding cassette sub-family C member 3 [mouse; UniProt B2RX12]',
    'B3A0R7': 'Methionine-rich protein [Lottia gigantea; UniProt B3A0R7]',
    'B5X0E4': 'Abcb5 - ATP-binding cassette sub-family B member 5 [mouse; UniProt B5X0E4]',
    'B8K1W2': 'Abcb11e - Bile salt export pump [dog; UniProt B8K1W2]',
    'D3ZCM3': 'Abcg4 - ATP-binding cassette subfamily G member 4 [rat; UniProt D3ZCM3]',
    'E7F6F7': 'Iron-sulfur clusters transporter ABCB7, mitochondrial [zebrafish; UniProt E7F6F7]',
    'E9PXX9': 'Slc28a1 - Sodium/nucleoside cotransporter 1 [mouse; UniProt E9PXX9]',
    'E9Q236': 'Abcc4 - ATP-binding cassette sub-family C member 4 [mouse; UniProt E9Q236]',
    'E9RBG1': 'abcC - ABC multidrug transporter C [Aspergillus fumigatus; UniProt E9RBG1]',
    'F1M3J4': 'Abcc4 - ATP-binding cassette subfamily C member 4 [rat; UniProt F1M3J4]',
    'F1NJ67': 'SLC46A1 - Proton-coupled folate transporter [chicken; UniProt F1NJ67]',
    'F2SG60': 'ABC multidrug transporter MDR3 [Trichophyton rubrum; UniProt F2SG60]',
    'F2SQT8': 'ABC multidrug transporter MDR5 [Trichophyton rubrum; UniProt F2SQT8]',
    'F8S296': 'ATG2 - Autophagy-related protein 2 [Arabidopsis; UniProt F8S296]',
    'G5EBN9': 'snf-3 - Sodium- and chloride-dependent betaine transporter [C. elegans; UniProt G5EBN9]',
    'G5EE72': 'mrp-5 - Multidrug resistance-associated protein 5 [C. elegans; UniProt G5EE72]',
    'H6TB12': 'mdr - Sophorolipid transporter [Starmerella bombicola; UniProt H6TB12]',
    'H9BZ66': 'ABCG1 - ABC transporter G family member 1 [Petunia hybrida; UniProt H9BZ66]',
    'O00264': 'PGRMC1 - Membrane-associated progesterone receptor component 1 [human; UniProt O00264]',
    'O02713': 'SLC22A2 - Solute carrier family 22 member 2 [pig; UniProt O02713]',
    'O08705': 'Slc10a1 - Hepatic sodium/bile acid cotransporter [mouse; UniProt O08705]',
    'O13973': 'mrp10 - Small ribosomal subunit protein mS37 [fission yeast; UniProt O13973]',
    'O15427': 'SLC16A3 - Monocarboxylate transporter 4 [human; UniProt O15427]',
    'O15438': 'ABCC3 - ATP-binding cassette sub-family C member 3 [human; UniProt O15438]',
    'O15439': 'ABCC4 - ATP-binding cassette sub-family C member 4 [human; UniProt O15439]',
    'O15440': 'ABCC5 - ATP-binding cassette sub-family C member 5 [human; UniProt O15440]',
    'O18735': 'ERBB2 - Receptor tyrosine-protein kinase erbB-2 [dog; UniProt O18735]',
    'O35114': 'Scarb2 - Lysosome membrane protein 2 [mouse; UniProt O35114]',
    'O35379': 'Abcc1 - ATP-binding cassette sub-family C member 1 [mouse; UniProt O35379]',
    'O35439': 'SLC16A1 - Monocarboxylate transporter 1 [Meriones unguiculatus; UniProt O35439]',
    'O35763': 'Msn - Moesin [rat; UniProt O35763]',
    'O35913': 'Slco1a4 - Solute carrier organic anion transporter family member 1A4 [rat; UniProt O35913]',
    'O35956': 'Slc22a6 - Solute carrier family 22 member 6 [rat; UniProt O35956]',
    'O42574': 'adrb1 - Beta-1 adrenergic receptor [African clawed frog; UniProt O42574]',
    'O42859': 'mrp2 - Small ribosomal subunit protein uS14m [fission yeast; UniProt O42859]',
    'O54715': 'Atp6ap1 - V-type proton ATPase subunit S1 [rat; UniProt O54715]',
    'O55519': 'Minor outer capsid protein P2 [Rice dwarf virus; UniProt O55519]',
    'O60706': 'ABCC9 - ATP-binding cassette sub-family C member 9 [human; UniProt O60706]',
    'O70594': 'Slc22a5 - Organic cation/carnitine transporter 2 [rat; UniProt O70594]',
    'O74907': 'Protein bcp1 [fission yeast; UniProt O74907]',
    'O75027': 'Iron-sulfur clusters transporter ABCB7, mitochondrial [human; UniProt O75027]',
    'O75751': 'SLC22A3 - Solute carrier family 22 member 3 [human; UniProt O75751]',
    'O80725': 'ABCB4 - ABC transporter B family member 4 [Arabidopsis; UniProt O80725]',
    'O88269': 'Abcc6 - ATP-binding cassette sub-family C member 6 [rat; UniProt O88269]',
    'O88397': 'Slco1a5 - Solute carrier organic anion transporter family member 1A5 [rat; UniProt O88397]',
    'O88909': 'Slc22a8 - Organic anion transporter 3 [mouse; UniProt O88909]',
    'O94956': 'SLCO2B1 - Solute carrier organic anion transporter family member 2B1 [human; UniProt O94956]',
    'O95477': 'Phospholipid-transporting ATPase ABCA1 [human; UniProt O95477]',
    'P05025': 'Sodium/potassium-transporting ATPase subunit alpha [Tetronarce californica; UniProt P05025]',
    'P05109': 'S100A8 - Protein S100-A8 [human; UniProt P05109]',
    'P06795': 'Abcb1b - ATP-dependent translocase ABCB1 [mouse; UniProt P06795]',
    'P08183': 'ATP-dependent translocase ABCB1 [human; UniProt P08183]',
    'P08793': 'XDH - Xanthine dehydrogenase [Calliphora vicina; UniProt P08793]',
    'P0DKC3': 'PGLP1A - Phosphoglycolate phosphatase 1A, chloroplastic [Arabidopsis; UniProt P0DKC3]',
    'P12687': "MRP7 - Large ribosomal subunit protein bL27m [baker's yeast; UniProt P12687]",
    'P12866': "STE6 - Alpha-factor-transporting ATPase [baker's yeast; UniProt P12866]",
    'P13568': 'MDR1 - Multidrug resistance protein 1 [Plasmodium falciparum; UniProt P13568]',
    'P13569': 'CFTR - Cystic fibrosis transmembrane conductance regulator [human; UniProt P13569]',
    'P14745': 'CD44 antigen [hamadryas baboon; UniProt P14745]',
    'P15308': 'Protein B4 [African clawed frog; UniProt P15308]',
    'P16444': 'DPEP1 - Dipeptidase 1 [human; UniProt P16444]',
    'P16876': 'MDR3 - Multidrug resistance protein 3 [Entamoeba histolytica; UniProt P16876]',
    'P16877': 'MDR4 - Multidrug resistance protein 4 [Entamoeba histolytica; UniProt P16877]',
    'P16970': 'Abcd3 - ATP-binding cassette sub-family D member 3 [rat; UniProt P16970]',
    'P17261': "ERS1 - Cystine transporter [baker's yeast; UniProt P17261]",
    'P21440': 'Phosphatidylcholine translocator ABCB4 [mouse; UniProt P21440]',
    'P21441': 'PGPA - Multidrug resistance protein [Leishmania tarentolae; UniProt P21441]',
    'P21447': 'Abcb1a - ATP-dependent translocase ABCB1 [mouse; UniProt P21447]',
    'P21449': 'PGY2 - Multidrug resistance protein 2 [Chinese hamster; UniProt P21449]',
    'P21958': 'Tap1 - Antigen peptide transporter 1 [mouse; UniProt P21958]',
    'P23174': 'Phosphatidylcholine translocator ABCB4 [Chinese hamster; UniProt P23174]',
    'P25439': 'brm - Brahma chromatin-remodeling complex ATPase subunit [fruit fly; UniProt P25439]',
    'P31596': 'Slc1a2 - Excitatory amino acid transporter 2 [rat; UniProt P31596]',
    'P32872': "ALD2 - Aldehyde dehydrogenase 2, mitochondrial [baker's yeast; UniProt P32872]",
    'P33300': "Mannosyl phosphorylinositol ceramide synthase SUR1 [baker's yeast; UniProt P33300]",
    'P33527': 'ABCC1 - ATP-binding cassette sub-family C member 1 [human; UniProt P33527]',
    'P35292': 'Rab17 - Ras-related protein Rab-17 [mouse; UniProt P35292]',
    'P36371': 'Tap2 - Antigen peptide transporter 2 [mouse; UniProt P36371]',
    'P41234': 'Abca2 - ATP-binding cassette sub-family A member 2 [mouse; UniProt P41234]',
    'P45844': 'ABCG1 - ATP-binding cassette sub-family G member 1 [human; UniProt P45844]',
    'P46720': 'Slco1a1 - Solute carrier organic anion transporter family member 1A1 [rat; UniProt P46720]',
    'P51670': 'Ccl9 - C-C motif chemokine 9 [mouse; UniProt P51670]',
    'P53007': 'SLC25A1 - Tricarboxylate transport protein, mitochondrial [human; UniProt P53007]',
    'P55061': 'TMBIM6 - Bax inhibitor 1 [human; UniProt P55061]',
    'P84813': 'Potamin-1 [Solanum tuberosum; UniProt P84813]',
    'P86410': 'Ralgapb - Ral GTPase-activating protein subunit beta [rat; UniProt P86410]',
    'Q00449': 'Mdr49 - Multidrug resistance protein homolog 49 [fruit fly; UniProt Q00449]',
    'Q00748': 'Mdr65 - Multidrug resistance protein homolog 65 [fruit fly; UniProt Q00748]',
    'Q00910': 'Slco2a1 - Solute carrier organic anion transporter family member 2A1 [rat; UniProt Q00910]',
    'Q04437': "YKU80 - ATP-dependent DNA helicase II subunit 2 [baker's yeast; UniProt Q04437]",
    'Q09427': 'ABCC8 - ATP-binding cassette sub-family C member 8 [hamster; UniProt Q09427]',
    'Q10589': 'BST2 - Bone marrow stromal antigen 2 [human; UniProt Q10589]',
    'Q13794': 'PMAIP1 - Phorbol-12-myristate-13-acetate-induced protein 1 [human; UniProt Q13794]',
    'Q15311': 'RALBP1 - RalA-binding protein 1 [human; UniProt Q15311]',
    'Q28689': 'ABCC2 - ATP-binding cassette sub-family C member 2 [rabbit; UniProt Q28689]',
    'Q3SZQ2': 'SLC22A17 - Solute carrier family 22 member 17 [bovine; UniProt Q3SZQ2]',
    'Q3ZAV1': 'Slc22a12 - Solute carrier family 22 member 12 [rat; UniProt Q3ZAV1]',
    'Q42093': 'ABCC2 - ABC transporter C family member 2 [Arabidopsis; UniProt Q42093]',
    'Q4GZT4': 'Broad substrate specificity ATP-binding cassette transporter ABCG2 [bovine; UniProt Q4GZT4]',
    'Q54BU4': 'abcB1 - ABC transporter B family member 1 [social amoeba; UniProt Q54BU4]',
    'Q54U44': 'abcC12 - ABC transporter C family member 12 [social amoeba; UniProt Q54U44]',
    'Q5T3U5': 'ABCC10 - ATP-binding cassette sub-family C member 10 [human; UniProt Q5T3U5]',
    'Q60754': 'Macrophage receptor MARCO [mouse; UniProt Q60754]',
    'Q7TMS5': 'Broad substrate specificity ATP-binding cassette transporter ABCG2 [mouse; UniProt Q7TMS5]',
    'Q7TSF7': 'Larp1 - La-related protein 1 [mouse; UniProt Q7TSF7]',
    'Q84K47': 'ABCA2 - ABC transporter A family member 2 [Arabidopsis; UniProt Q84K47]',
    'Q84M24': 'ABCA1 - ABC transporter A family member 1 [Arabidopsis; UniProt Q84M24]',
    'Q8CHI8': 'Ep400 - E1A-binding protein p400 [mouse; UniProt Q8CHI8]',
    'Q8LPT1': 'ABCB6 - ABC transporter B family member 6 [Arabidopsis; UniProt Q8LPT1]',
    'Q8MIB3': 'Broad substrate specificity ATP-binding cassette transporter ABCG2 [pig; UniProt Q8MIB3]',
    'Q8VZZ4': 'ABCC6 - ABC transporter C family member 6 [Arabidopsis; UniProt Q8VZZ4]',
    'Q92886': 'NEUROG1 - Neurogenin-1 [human; UniProt Q92886]',
    'Q92887': 'ABCC2 - ATP-binding cassette sub-family C member 2 [human; UniProt Q92887]',
    'Q96J66': 'ABCC11 - ATP-binding cassette sub-family C member 11 [human; UniProt Q96J66]',
    'Q99466': 'NOTCH4 - Neurogenic locus notch homolog protein 4 [human; UniProt Q99466]',
    'Q9FHF1': 'ABCB7 - ABC transporter B family member 7 [Arabidopsis; UniProt Q9FHF1]',
    'Q9H172': 'ABCG4 - ATP-binding cassette sub-family G member 4 [human; UniProt Q9H172]',
    'Q9LHD1': 'ABCB15 - ABC transporter B family member 15 [Arabidopsis; UniProt Q9LHD1]',
    'Q9NGP5': 'abcG2 - ABC transporter G family member 2 [social amoeba; UniProt Q9NGP5]',
    'Q9P983': 'deleted [; UniProt Q9P983]',
    'Q9QYV9': 'Asic4 - Acid-sensing ion channel 4 [rat; UniProt Q9QYV9]',
    'Q9SYI3': 'ABCB5 - ABC transporter B family member 5 [Arabidopsis; UniProt Q9SYI3]',
    'Q9U539': 'oct-1 - Organic cation transporter 1 [C. elegans; UniProt Q9U539]',
    'Q9UKJ4': 'Scavenger receptor cysteine-rich domain-containing protein DMBT1 [human; UniProt Q9UKJ4]',
    'Q9UNQ0': 'Broad substrate specificity ATP-binding cassette transporter ABCG2 [human; UniProt Q9UNQ0]',
}

NCBI_GENE_NAMES = {
    '19': 'ABCA1 - ATP binding cassette subfamily A member 1 [human; NCBI Gene 19]',
    '20': 'ABCA2 - ATP binding cassette subfamily A member 2 [human; NCBI Gene 20]',
    '368': 'ABCC6 - ATP binding cassette subfamily C member 6 [human; NCBI Gene 368]',
    '480': 'ATP1A4 - ATPase Na+/K+ transporting subunit alpha 4 [human; NCBI Gene 480]',
    '1080': 'CFTR - CF transmembrane conductance regulator [human; NCBI Gene 1080]',
    '1244': 'ABCC2 - ATP binding cassette subfamily C member 2 [human; NCBI Gene 1244]',
    '4194': 'MDM4 - MDM4 regulator of p53 [human; NCBI Gene 4194]',
    '4363': 'ABCC1 - ATP binding cassette subfamily C member 1 (ABCC1 blood group) [human; NCBI Gene 4363]',
    '4625': 'MYH7 - myosin heavy chain 7 [human; NCBI Gene 4625]',
    '4948': 'OCA2 - OCA2 melanosomal transmembrane protein [human; NCBI Gene 4948]',
    '5243': 'ABCB1 - ATP binding cassette subfamily B member 1 [human; NCBI Gene 5243]',
    '5244': 'ABCB4 - ATP binding cassette subfamily B member 4 [human; NCBI Gene 5244]',
    '6567': 'SLC16A2 - solute carrier family 16 member 2 [human; NCBI Gene 6567]',
    '6833': 'ABCC8 - ATP binding cassette subfamily C member 8 [human; NCBI Gene 6833]',
    '8647': 'ABCB11 - ATP binding cassette subfamily B member 11 [human; NCBI Gene 8647]',
    '8712': 'PAGE1 - PAGE family member 1 [human; NCBI Gene 8712]',
    '8714': 'ABCC3 - ATP binding cassette subfamily C member 3 [human; NCBI Gene 8714]',
    '9376': 'SLC22A8 - solute carrier family 22 member 8 [human; NCBI Gene 9376]',
    '9429': 'ABCG2 - ATP binding cassette subfamily G member 2 (JR blood group) [human; NCBI Gene 9429]',
    '9619': 'ABCG1 - ATP binding cassette subfamily G member 1 [human; NCBI Gene 9619]',
    '10057': 'ABCC5 - ATP binding cassette subfamily C member 5 [human; NCBI Gene 10057]',
    '10257': 'ABCC4 - ATP binding cassette subfamily C member 4 (PEL blood group) [human; NCBI Gene 10257]',
    '10928': 'RALBP1 - ralA binding protein 1 [human; NCBI Gene 10928]',
    '11194': 'ABCB8 - ATP binding cassette subfamily B member 8 [human; NCBI Gene 11194]',
    '64137': 'ABCG4 - ATP binding cassette subfamily G member 4 [human; NCBI Gene 64137]',
    '64756': 'ATPAF1 - ATP synthase mitochondrial F1 complex assembly factor 1 [human; NCBI Gene 64756]',
    '85320': 'ABCC11 - ATP binding cassette subfamily C member 11 [human; NCBI Gene 85320]',
    '89845': 'ABCC10 - ATP binding cassette subfamily C member 10 [human; NCBI Gene 89845]',
    '94160': 'ABCC12 - ATP binding cassette subfamily C member 12 [human; NCBI Gene 94160]',
    '283871': 'PGP - phosphoglycolate phosphatase [human; NCBI Gene 283871]',
    '340273': 'ABCB5 - ATP binding cassette subfamily B member 5 [human; NCBI Gene 340273]',
    '644079': 'BCRP1 - BCR pseudogene 1 [human; NCBI Gene 644079]',
}

# Names are from UMLS-indexed BioThings services or public UMLS cross-references.
# Missing concepts remain raw because UMLS content retrieval requires credentials.
UMLS_NAMES = {
    "C0069906": "P-glycoproteins [UMLS C0069906]",
    "C0070312": "pentenomycin II [UMLS C0070312]",
    "C0242738": "ATP-binding cassette transporters [UMLS C0242738]",
    "C0290741": "multidrug resistance-associated proteins [UMLS C0290741]",
    "C0376622": "ABCB1 [UMLS C0376622]",
    "C0525411": "monocarboxylic acid transporters [UMLS C0525411]",
    "C0596902": "membrane transport proteins [UMLS C0596902]",
    "C0919458": "ABCC1 [UMLS C0919458]",
    "C0949791": "organic anion transporting polypeptides [UMLS C0949791]",
    "C0969703": "organic anion transporters [UMLS C0969703]",
    "C1332002": "ABCG2 [UMLS C1332002]",
    "C1418661": "PLXNA2 - plexin A2 [UMLS C1418661]",
    "C1422070": "PGPEP1 - pyroglutamyl-peptidase I [UMLS C1422070]",
    "C4277650": "ATP-binding cassette transporter subfamily G [UMLS C4277650]",
}

GENE_VARIANT_NAMES = {
    "GeneVar:482T": "ABCG2 R482T variant",
    "GeneVar:R482G": "ABCG2 R482G variant",
    "GeneVar:ABCG2 (R482)": "ABCG2 R482 variant",
    "GeneVar:ABCG2-G482": "ABCG2 R482G variant",
    "GeneVar:ABCG2 p.F431L": "ABCG2 F431L variant",
    "GeneVar:ABCG2 p.R482G": "ABCG2 R482G variant",
    "GeneVar:ABCG2 p.R482T": "ABCG2 R482T variant",
    "GeneVar:ABCG2 p.T469A": "ABCG2 T469A variant",
    "GeneVar:ABCG2 p.Y581S": "ABCG2 Y581S variant",
    "GeneVar:ABCC4 p.E757K": "ABCC4 E757K variant",
    "GeneVar:MRP3-Arg1297His": "ABCC3 Arg1297His variant",
    "GeneVar:Abcb1a/1b;Abcg2-/-": "Abcb1a/Abcb1b/Abcg2 knockout",
}

_UNIPROT = re.compile(r"UniProt:([A-Z0-9]+)")
_NCBI_GENE = re.compile(r"NCBIGene:(\d+)")
_UMLS = re.compile(r"UMLS:(C\d+)")


def canonical_transporter_identifier(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    if raw in GENE_VARIANT_NAMES:
        return GENE_VARIANT_NAMES[raw]
    raw = _UNIPROT.sub(lambda match: UNIPROT_NAMES.get(match[1], match[0]), raw)
    raw = _NCBI_GENE.sub(lambda match: NCBI_GENE_NAMES.get(match[1], match[0]), raw)
    canonical = _UMLS.sub(lambda match: UMLS_NAMES.get(match[1], match[0]), raw)
    if any(marker in canonical for marker in ("UniProt:", "NCBIGene:", "UMLS:", "GeneVar:", "SMILES:")):
        return None
    canonical = re.sub(r"\s*\[; (?:UniProt [A-Z0-9]+|NCBI Gene \d+)\]", "", canonical)
    canonical = re.sub(r"; (?:UniProt [A-Z0-9]+|NCBI Gene \d+)\]", "]", canonical)
    canonical = re.sub(r"\s*\[UMLS C\d+\]", "", canonical)
    return canonical


def transporter_identifier_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    raw = record.get("transporter_identifier")
    canonical = canonical_transporter_identifier(raw)
    return {
        "normalized_transporter_identifier": canonical,
        "transporter_identifier_mapping_status": (
            "not_available"
            if canonical is None and not str(raw or "").strip()
            else "unresolved_source_identifier"
            if canonical is None
            else "source_semantic_label"
            if canonical == str(raw).strip()
            else "mapped"
        ),
        "transporter_identifier_policy_version": TRANSPORTER_IDENTIFIER_VERSION,
    }


__all__ = [
    "TRANSPORTER_IDENTIFIER_VERSION",
    "canonical_transporter_identifier",
    "transporter_identifier_fields",
]
