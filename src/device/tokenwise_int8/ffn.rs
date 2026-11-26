//! Layout of the packed Up/Gate weights.
use furiosa_opt_std::prelude::*;

axes![Lu = 30720];
pub(crate) type UCl = m![Lu / 512 % 2];
