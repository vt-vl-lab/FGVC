# [ECCV 2020] Flow-edge Guided Video Completion

### [[Paper](https://arxiv.org/abs/2009.01835)] [[Project Website](http://chengao.vision/FGVC/)] [[Google Colab](#) (coming soon)]

<p align='center'>
<img src='http://chengao.vision/FGVC/files/FGVC_teaser.png' width='900'/>
</p>

We present a new flow-based video completion algorithm. Previous flow completion methods are often unable to retain the sharpness of motion boundaries. Our method first extracts and completes motion edges, and then uses them to guide piecewise-smooth flow completion with sharp edges. Existing methods propagate colors among local flow connections between adjacent frames. However, not all missing regions in a video can be reached in this way because the motion boundaries form impenetrable barriers. Our method alleviates this problem by introducing non-local flow connections to temporally distant frames, enabling propagating video content over motion boundaries. We validate our approach on the DAVIS dataset. Both visual and quantitative results show that our method compares favorably against the state-of-the-art algorithms.
<br/>

**[ECCV 2020] Flow-edge Guided Video Completion**
<br/>
[Chen Gao](http://chengao.vision), [Ayush Saraf](#), [Jia-Bin Huang](https://filebox.ece.vt.edu/~jbhuang/), and [Johannes Kopf](https://johanneskopf.de/)
<br/>
In European Conference on Computer Vision (ECCV), 2020

## Note from Chen

This is a beta version of the code. I will work on making the official code available (tentative date: late-Nov).


## Prerequisites

- Linux (tested on CentOS Linux release 7.4.1708)
- Anaconda
- Python 3.6
- PyTorch 0.4.0 (for DeepFill). Please set up another environment `FGVC` according to [this](https://github.com/nbei/Deep-Flow-Guided-Video-Inpainting#install--requirements).
- PyTorch 1.6.0 (for RAFT). Please set up an environment `raft` according to [this](https://github.com/princeton-vl/RAFT#requirements).

## Quick start

Always activate `raft` first. Run the code until you see a warning: 'Please switch to Pytorch 0.4.0'. Deactivate `raft`, activate `FGVC`, and run the code again.

- Object removal:
```bash
cd tool
python video_completion.py \
       --mode object_removal \
       --path ../data/tennis \
       --path_mask ../data/tennis_mask \
       --outroot ../result/tennis_removal \
       --seamless
```

- FOV extrapolation:
```bash
cd tool
python video_completion.py \
       --mode video_extrapolation \
       --path ../data/tennis \
       --outroot ../result/tennis_extrapolation \
       --H_scale 2 \
       --W_scale 2 \
       --seamless
```

You can remove the `--seamless` flag for a faster processing time.


## License
This work is licensed under MIT License. See [LICENSE](LICENSE) for details.

If you find this code useful for your research, please consider citing the following paper:

	@inproceedings{Gao-ECCV-FGVC,
	    author    = {Gao, Chen and Saraf, Ayush and Huang, Jia-Bin and Kopf, Johannes},
	    title     = {Flow-edge Guided Video Completion},
	    booktitle = {European Conference on Computer Vision},
	    year      = {2020}
	}

## Acknowledgments
- Our flow edge completion network builds upon [EdgeConnect](https://github.com/knazeri/edge-connect).
