# Smart EC: Program-Building for Name Mapping
The original 3rd place solution to the 2018 ICCAD Problem A: Smart-EC.

After fixing a bug (due to an uncleared definition of 1-to-1 mapping in the problem definition), the program has similar performances as the 1st place.

The problem definition is [here](https://github.com/Daikon-Sun/Smart-EC/blob/master/ProbDef.pdf).

## Required Packages and Versions
- python3 >= 3.4.5

## File Descriptions
- main.py: the main program
- testcases: all cases in the contest
- outputs: all generated python scripts
- ProbDef.pdf: problem definition.

## Usage
```
python3 main.py input_json output_py
```

## Example
Check out the script `run.sh`, which needs two arguments.

The first argument is the id of the testcase, and the second argument is the name of the output python script.

## Results
| Testcases | Size of Script |
|:-:|---:|
| 0 | 95 |
| 1 | 221 | 
| 2 | 221 |
| 3 | 167 |
| 4 | 167 |
| 5 | 294 |
| 6 | 181 |
| 7 | 181 |
| 8 | 1222 |
| 9 | 99650  |
| 10 | 133158 |
| 11 | 137310 |
| 12 | 153663 |
| 13 | 163834 |
| 14 | 167 |
| 15 | 153233 |
| 16 | 95934 |
| 17 | Error |
| 18 | 166 |
| 19 | 98034 |
| 20 | 93024 |
| 21 | 117830 |
| 22 | 121549 |
| 23 | 103188 |

## Additional Informations
The program needs 64GB of memory!

This is the original constraint of the contest.
