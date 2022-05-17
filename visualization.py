import argparse
import warnings

from pylab import *
from torch.utils.data import DataLoader

from dataset import CelebABalance as CelebA
from utils import *

warnings.filterwarnings("ignore")

attr_list = ('5_o_Clock_Shadow,Arched_Eyebrows,Attractive,Bags_Under_Eyes,Bald,Bangs,Big_Lips,Big_Nose,'
             'Black_Hair,Blond_Hair,Blurry,Brown_Hair,Bushy_Eyebrows,Chubby,Double_Chin,Eyeglasses,Goatee,Gray_Hair,'
             'Heavy_Makeup,High_Cheekbones,Male,Mouth_Slightly_Open,Mustache,Narrow_Eyes,No_Beard,Oval_Face,Pale_Skin,'
             'Pointy_Nose,Receding_Hairline,Rosy_Cheeks,Sideburns,Smiling,Straight_Hair,Wavy_Hair,Wearing_Earrings,'
             'Wearing_Hat,Wearing_Lipstick,Wearing_Necklace,Wearing_Necktie,Young'
             ).split(',')

attr_dict = {}
for i, attr in enumerate(attr_list):
    attr_dict[attr] = i

insufficient_attr_list = '5_o_Clock_Shadow,Goatee,Mustache,Sideburns,Wearing_Necktie'.split(',')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default="../data/celeba/celeba.hdf5")
    parser.add_argument('--result-dir', type=str, default='results')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--resume', action="store_true", default=False)
    parser.add_argument('--domain-attrs', type=str, default='Male')
    parser.add_argument('--target-attrs', type=str, default='Blond_Hair')
    parser.add_argument('--exp-name', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=2e-5)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--arch', type=str, default="resnet18", choices=["resnet18", "resnet20s", "resnet9"])
    parser.add_argument('--evaluate', action="store_true")
    parser.add_argument('--total-num', default=None, type=int)

    parser.add_argument('--gv', type=float, default=0.0, help="Gaussian Noise")
    parser.add_argument('--gr', type=float, default=0.0)
    parser.add_argument('--minor-ratio', type=float, default=0.2, help="the ratio of minority group")

    args = parser.parse_args()

    assert args.target_attrs in attr_list
    assert args.domain_attrs in attr_list

    return args


def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    setup_seed(args.seed)

    # Sanity Check!
    assert args.data_dir is not None

    image_size = 224
    transform_train, transform_test = get_transform(image_size=image_size)

    num_class = 2

    test_set = CelebA(root=args.data_dir, target_attr=args.target_attrs,
                      transform=transform_test, split="test", gaussian_aug_ratio=0.0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)

    pbar = tqdm(test_loader, total=len(test_loader), ncols=120)
    bm, bf, nm, nf = False, False, False, False
    for x, (y, d) in pbar:
        x, y, d = x.to(device), y.to(device), d.to(device)

        for image, hair, male in zip(x, y, d):
            if not bm and hair and male:
                plt.axis('off')
                plt.tight_layout()
                plt.imshow(image.permute(1, 2, 0))
                plt.savefig("graphs/blond_male.png", bbox_inches='tight', pad_inches=0)
                plt.show()

                for gv in [1, 0.1, 0.01, 0.001]:
                    image_noise = image + torch.randn_like(image) * np.sqrt(gv)
                    image_noise = torch.clip(image_noise, min=0.0, max=1.0)

                    plt.axis('off')
                    plt.tight_layout()
                    plt.imshow(image_noise.permute(1, 2, 0))
                    plt.savefig(f"graphs/blond_male_{gv}.png", bbox_inches='tight', pad_inches=0)
                    plt.show()

                bm = True

            if not bf and hair and not male:
                plt.axis('off')
                plt.tight_layout()
                plt.imshow(image.permute(1, 2, 0))
                plt.savefig("graphs/blond_female.png", bbox_inches='tight', pad_inches=0)
                plt.show()
                bf = True

            if not nm and not hair and male:
                plt.axis('off')
                plt.tight_layout()
                plt.imshow(image.permute(1, 2, 0))
                plt.savefig("graphs/nonblond_male.png", bbox_inches='tight', pad_inches=0)
                plt.show()
                nm = True

            if not nf and not hair and not male:
                plt.axis('off')
                plt.tight_layout()
                plt.imshow(image.permute(1, 2, 0))
                plt.savefig("graphs/nonblond_female.png", bbox_inches='tight', pad_inches=0)
                plt.show()
                nf = True

            if nf and nm and bf and bm:
                sys.exit()


if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)
