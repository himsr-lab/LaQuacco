"""
Copyright 2023 The Regents of the University of Colorado

This file is part of LaQuacco.

LaQuacco is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

LaQuacco is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

Author:     Christian Rickert <christian.rickert@cuanschutz.edu>
Group:      Human Immune Monitoring Shared Resource (HIMSR)
            University of Colorado, Anschutz Medical Campus

Title:      LaQuacco
Summary:    Laboratory Quality Control v2.0 (2024-10-17)
DOI:        # TODO
URL:        https://github.com/himsr-lab/LaQuacco
"""

import pytest


def fail():
    raise SystemExit(1)


def test_fail():
    with pytest.raises(SystemExit):
        fail()
