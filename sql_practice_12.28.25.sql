create database practice_dec;
use practice_dec;

SET SQL_SAFE_UPDATES = 0;

drop table t1;
create table t1 (
	id int primary key,
    fname varchar(50),
    lname varchar(50),
    age int);

drop table t3;
create table t3 (
	petid varchar(4) primary key,
    petname varchar(50),
    pettype varchar(50),
    constraint petid CHECK (petid LIKE '%p')
    );

drop table t2;
create table t2 (
	id int,
    petid varchar(50),
	foreign key (id) references t1(id),
    foreign key (petid) references t3(petid),
    constraint petid CHECK (petid LIKE '%p'));
    
select * from t1;
select * from t2;
select * from t3;

insert into t1 (id, fname, lname, age) 
values (0, "Joe", "deBlois", 23), 
		(1, "Owen", "Stanforth", 29), 
		(2, "Mike", "deBlois", 50), 
        (3, "Beth", "deBlois", 50),
        (4, "John", "Doe", 68),
        (5, "Jane", "Stanforth", 8),
        (6, "Will", "John", 13);

insert into t3 (petid, petname, pettype)
values ("0p", "Oreo", "dog"),
		("1p", "Fiver", "rabbit"),
        ("2p", "Spot", "dog"),
        ("3p", "Cheddar", "cat"),
        ("4p", "Cube", "fish"),
        ("5p", "Lucy", "dog");
        
insert into t2 (id, petid)
values (0, "0p"),
		(0, "1p"),
        (2, "0p"),
        (3, "0p"),
        (4, "2p"),
        (4, "3p"),
        (6, "2p"),
        (6, "4p"),
        (6, "5p"),
        (5, "2p");

select * from t1;
select * from t2;
select * from t3;


-- practice!

-- 1) show the names of all owners
select fname from t1;
select t1.fname from t1 right join t2 on t1.id = t2.id;

-- only once?
select distinct t1.fname from t1 right join t2 on t1.id = t2.id;

-- 2) show the names of all owners >=50
select fname, lname from t1 where age >= 50;

-- 3) show the names of owners and their associated pets
select distinct t1.fname, t3.petname
from t2 
left join t1 on t2.id = t1.id
left join t3 on t2.petid = t3.petid;

-- 4) show how many pets each owner has
select t1.fname, count(t2.petid) as numpets
from t2
right join t1 on t1.id = t2.id
group by t1.fname;

-- 5) show the name of each pet owned by >1 person
select t3.petname, count(t2.petid)
from t2
join t3 on t2.petid = t3.petid
group by t3.petid, t3.petname
having count(t2.petid) > 1;

-- 6) show all first names with the same last name (self join)
select a.lname, count(distinct a.id) as num_people
from t1 as a
join t1 as b
on a.lname = b.lname
and a.id != b.id
group by a.lname;

select lname, count(*)
from t1
group by lname;

-- 7) how many pet types
select count(distinct pettype)
from t3;

-- 8) name all pet types
select distinct pettype
from t3;

-- 9) change all "rabbit" to "bunny"
update t3
set pettype = "bunny"
where pettype = "rabbit";

select * from t3;

-- 10) alter table 1 to add column "favcolor";blue if last name deBlois, red if last name Stanforth, else yellow
alter table t1
add column favcolor varchar(50);
alter table t1
add constraint 
check (favcolor in ("blue", "red", "yellow"));

update t1
set favcolor = 
	case
when lname = "deBlois" then "blue"
when lname = "Stanforth" then "red"
else "yellow"
	end;

select * from t1;

-- 11) list all pet names next to unique owner's favorite colors

-- step 1: list all pets per person
select t3.petname , t1.favcolor
from t2
right join t3 on t3.petid = t2.petid
right join t1 on t1.id = t2.id
where t3.petname is not null
order by t3.petname, t1.favcolor;