# sc-adapted-scripts

A collection of modified open-source tools, adapted for my specific needs and workflows. Includes enhancements, bug fixes, and personal customizations. This repository contains Python scripts that I have adapted from various sources. These modifications often involve adding new features, improving efficiency, or tailoring them to specific tasks.

## Overview

Here you will find various utilities that have been tweaked, enhanced, or customized from their original versions. The modifications may include:

* Adding new features or functionalities
* Fixing bugs or addressing specific issues I encountered
* Optimizing performance or efficiency
* Integrating with other tools or workflows
* Simplifying usage for my particular use cases

## Contents

* `[scverse_related/Palantir/Run_magic_imputation.py]`: [Avoid MemoryError]

When using the `run_magic_imputation` function in Palantir with a large number of cells, I encountered memory issues during parallel processing. To address the "MemoryError: Unable to allocate 118. GiB for an array with shape (126007, 126007) and data type float64" issue (related to https://github.com/dpeerlab/Palantir/issues/34#issuecomment-632963651), I implemented a solution to output the parallel computation results to disk. This avoids the memory-intensive conversion between sparse and dense matrices in memory. This approach has been tested successfully with datasets of up to 500,000 cells, requiring approximately 200GB of disk space.
The original function can be found at: https://github.com/dpeerlab/Palantir/blob/5fe3b46043dd32c30942ae297071bd9c71794ac1/src/palantir/utils.py#L586

* ...


## Original Sources

Where applicable, I have tried to maintain references to the original sources of these tools. You can find links or acknowledgements within the individual tool's directory or documentation. I respect the licenses of the original projects and aim to use and share these modifications responsibly.

## Usage

Usually you need to install the original software into your environment and replace the original functions with monkey patch methods.
